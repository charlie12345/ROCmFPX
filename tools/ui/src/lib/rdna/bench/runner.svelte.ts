/**
 * RDNA-FAST bench runner: the owner's bench, run from the browser against one loaded model.
 *
 *   1 agent session 1 of 2   agentic_bench.py (streaming, client-timed TTFT and decode)
 *   2 code edits             edit_gate.py (server timings)
 *   3 decode answers         readme_bench.py, nine prose + nine code answers
 *   4 agent session 2 of 2
 *
 * Every request runs at priority xhigh (owner, 2026-11-03): other requests are held (they trickle
 * one pass every 2 s), so each step starts at once. While a request runs the runner polls /slots
 * every second and counts polls that saw another slot busy, held ones included (agentic_bench.py
 * --slots), and shows how many are held; after each step it looks for requests of other priorities
 * in /requests. Such a run is kept but marked. Steps the model's context cannot hold are skipped.
 * In router mode the benched model runs alone (owner, 2026-11-03): the other loaded models are
 * unloaded first and loaded back when the bench ends, however it ends; one that cannot be unloaded,
 * or that a request loads again mid-bench, marks the run.
 * The run lands in localStorage under the model's id. The runner is a module singleton, so a run
 * goes on while the user switches pages.
 */

import { modelQuery, rdnaGet } from '../api';
import { BENCH_RUNS_CAP } from '../constants';
import { normalizeRecord } from '../history';
import { readPart, writePart } from '../storage';
import type { RdnaModelEntry, RdnaSlot } from '../types';
import { readmeRow, runValues } from './metrics';
import {
	type BenchFixtures,
	buildSession,
	type ChatMessage,
	CODE,
	EDIT_MAX_TOKENS,
	editCases,
	fits,
	loadFixtures,
	PROSE,
	README_MAX_TOKENS,
	sessionNonce,
	TOKENS_EDIT_MAX,
	TOKENS_SESSION_MAX,
	verbatimShare
} from './scenario';
import type { BenchRun, EditRow, ReadmeCall, SessionResult, SessionStep } from './types';
import { base } from '$app/paths';
import { getJsonHeaders } from '$lib/utils/api-headers';

export type BenchPhase = 'idle' | 'loading' | 'running' | 'done' | 'stopped' | 'error';

/** The priority every bench request carries. */
export const BENCH_PRIORITY = 'xhigh';

export interface BenchTarget {
	modelId: string;
	routerMode: boolean;
	/** Context of one slot (the bench's longest prompt must fit). */
	nCtx: number | null;
	build: string | null;
	modelLabel: string;
}

const REQUEST_TIMEOUT_MS = 3600 * 1000;

class BenchError extends Error {}

/** Other requests on the model while one bench request runs: busy slots besides the bench's own. */
export function othersOnModel(slots: readonly RdnaSlot[]): number {
	return Math.max(0, slots.filter((s) => s.is_processing).length - 1);
}

/** " · 1 other request held" while the bench holds others. */
export function heldText(n: number): string {
	return n > 0 ? ` · ${n} other ${n === 1 ? 'request' : 'requests'} held` : '';
}

function isAbort(e: unknown): boolean {
	return e instanceof DOMException && e.name === 'AbortError';
}

function sleep(ms: number, signal: AbortSignal): Promise<void> {
	return new Promise((resolve, reject) => {
		const t = setTimeout(resolve, ms);

		signal.addEventListener(
			'abort',
			() => {
				clearTimeout(t);
				reject(new DOMException('Aborted', 'AbortError'));
			},
			{ once: true }
		);
	});
}

/** How long the bench waits for the other models to unload. */
const UNLOAD_WAIT_MS = 60_000;

/** Read the saved runs of a model. */
export function loadBenchRuns(modelId: string): BenchRun[] {
	const v = readPart<BenchRun[]>(modelId, 'bench');

	return Array.isArray(v) ? v.filter((r) => r && typeof r.t === 'number' && r.values) : [];
}

function saveBenchRun(run: BenchRun): void {
	const runs = loadBenchRuns(run.modelId);

	runs.push(run);
	runs.sort((a, b) => a.t - b.t);
	writePart(run.modelId, 'bench', runs.slice(-BENCH_RUNS_CAP));
}

interface StepPlan {
	id: 'session1' | 'edits' | 'decode' | 'session2';
	name: string;
	requests: number;
}

/** What the running bench has measured so far. */
interface LiveParts {
	sessions: SessionResult[];
	/** The session running now, its finished requests. */
	current: SessionStep[];
	edits: EditRow[];
	prose: ReadmeCall[];
	code: ReadmeCall[];
}

function emptyParts(): LiveParts {
	return { code: [], current: [], edits: [], prose: [], sessions: [] };
}

class BenchRunner {
	done = $state(0);
	/** A stopped or failed run kept what it had measured. */
	keptIncomplete = $state(false);
	message = $state('');
	modelId = $state<string | null>(null);
	modelLabel = $state('');
	/** Other requests on the model at the last poll (held while the bench runs). */
	othersHeld = $state(0);
	overlapped = $state(false);
	/**
	 * The run in progress as measured so far, refreshed after every request, so the speed card and
	 * the bench chart fill in while it runs (it counts as the newest run). Null when none runs.
	 */
	partial = $state.raw<BenchRun | null>(null);
	phase = $state<BenchPhase>('idle');
	/** What the preparation does now (unloading the other models), shown instead of the prompts. */
	prep = $state('');
	/** The models the bench unloaded are being loaded back. */
	restoring = $state(false);
	/** Bumped when a run is saved, so the page reloads its runs. */
	savedAt = $state(0);
	stepCount = $state(0);
	stepIndex = $state(0);
	stepName = $state('');
	total = $state(1);

	private abort: AbortController | null = null;
	private parts: LiveParts = emptyParts();
	private skipped: string[] = [];
	private target: BenchTarget | null = null;

	/** A finished run left something to look at: it is done, or it kept an incomplete run. */
	get hasResults(): boolean {
		return (
			this.phase === 'done' ||
			((this.phase === 'stopped' || this.phase === 'error') && this.keptIncomplete)
		);
	}

	get label(): string {
		const text = this.phaseText();

		return this.restoring && !this.running ? `${text} · loading the other models back` : text;
	}

	get pct(): number {
		if (this.phase === 'done') return 100;

		return Math.max(0, Math.min(100, Math.round((100 * this.done) / Math.max(1, this.total))));
	}

	get running(): boolean {
		return this.phase === 'loading' || this.phase === 'running';
	}

	get visible(): boolean {
		return this.phase !== 'idle';
	}

	close(): void {
		if (!this.running) this.phase = 'idle';
	}

	async start(target: BenchTarget): Promise<void> {
		if (this.running) return;

		this.target = target;
		this.modelId = target.modelId;
		this.modelLabel = target.modelLabel;
		this.abort = new AbortController();
		this.overlapped = false;
		this.othersHeld = 0;
		this.message = '';
		this.done = 0;
		this.stepIndex = 0;
		this.keptIncomplete = false;
		this.parts = emptyParts();
		this.partial = null;
		this.prep = '';
		this.phase = 'loading';

		const signal = this.abort.signal;
		// filled as each model unloads, so a stop mid-way still loads back the ones already gone
		const unloaded: string[] = [];

		try {
			const fx = await loadFixtures(signal);
			const plan = this.plan(target.nCtx);

			this.skipped = this.skippedSteps(target.nCtx);

			if (plan.length === 0) {
				throw new BenchError("the model's context is too small for the bench");
			}

			if (target.routerMode) await this.unloadOthers(signal, unloaded);

			this.stepCount = plan.length;
			this.total = plan.reduce((s, p) => s + p.requests, 0);

			for (let i = 0; i < plan.length; i++) {
				const step = plan[i];

				this.phase = 'running';
				this.stepIndex = i + 1;
				this.stepName = step.name;

				const t0 = Date.now();

				if (step.id === 'session1' || step.id === 'session2') {
					await this.runSession(fx, signal);
				} else if (step.id === 'edits') {
					await this.runEdits(fx, signal);
				} else if (step.id === 'decode') {
					for (const p of PROSE) {
						this.parts.prose.push(await this.readmeCall(p, README_MAX_TOKENS, signal));
						this.publish();
					}

					for (const p of CODE) {
						this.parts.code.push(await this.readmeCall(p, README_MAX_TOKENS, signal));
						this.publish();
					}
				}

				if (await this.foreignRequests(t0, Date.now())) this.overlapped = true;
			}

			// a request for an unloaded model loads it again: the GPU was not the bench's alone
			if (unloaded.length > 0) {
				const loaded = await this.loadedModels();

				if (unloaded.some((id) => loaded.includes(id))) this.overlapped = true;
			}

			const run = this.snapshot();

			this.overlapped = run.overlapped;
			saveBenchRun(run);
			this.partial = null;
			this.savedAt = Date.now();
			this.phase = 'done';
		} catch (e) {
			// keep what the run measured, marked as incomplete
			const run = this.snapshot({ incomplete: true });

			if (Object.keys(run.values).length > 0) {
				saveBenchRun(run);
				this.keptIncomplete = true;
				this.savedAt = Date.now();
			}

			this.partial = null;

			if (isAbort(e) || signal.aborted) {
				this.phase = 'stopped';
			} else {
				this.message = e instanceof Error ? e.message : String(e);
				this.phase = 'error';
			}
		} finally {
			this.abort = null;
			this.othersHeld = 0;
			this.prep = '';

			if (unloaded.length > 0) void this.loadBack(unloaded);
		}
	}

	stop(): void {
		if (this.running) {
			this.abort?.abort();
			this.phase = 'stopped';
		} else {
			this.phase = 'idle';
		}
	}

	private body(extra: Record<string, unknown>): string {
		return JSON.stringify({ model: this.target?.modelId, priority: BENCH_PRIORITY, ...extra });
	}

	/** Did requests of another priority than low arrive while the step ran? */
	private async foreignRequests(from: number, to: number): Promise<boolean> {
		const res = await rdnaGet<{ requests?: unknown[] }>('/requests', this.query());

		if (!res.ok || !res.data || !Array.isArray(res.data.requests)) return false;

		return res.data.requests
			.map(normalizeRecord)
			.some(
				(r) =>
					r !== null &&
					r.t_arrival_ms >= from &&
					r.t_arrival_ms <= to &&
					(r.priority ?? 'default') !== BENCH_PRIORITY
			);
	}

	private async getSlots(signal?: AbortSignal): Promise<RdnaSlot[] | null> {
		const res = await rdnaGet<RdnaSlot[]>('/slots', this.query(), signal);

		if (res.status === 401) throw new BenchError('the API key was refused');

		return res.ok && Array.isArray(res.data) ? res.data : null;
	}

	private linkSignal(signal: AbortSignal): AbortSignal {
		return AbortSignal.any([signal, AbortSignal.timeout(REQUEST_TIMEOUT_MS)]);
	}

	/** Load back the models the bench unloaded, whichever way it ended (not cancellable). */
	private async loadBack(ids: string[]): Promise<void> {
		this.restoring = true;

		try {
			// one a request already loaded again answers "already running": nothing to do
			for (const id of ids) await this.modelCall('/models/load', id);
		} finally {
			this.restoring = false;
		}
	}

	/** The router's loaded (or loading) models. */
	private async loadedModels(signal?: AbortSignal): Promise<string[]> {
		const res = await rdnaGet<{ data?: RdnaModelEntry[] }>('/v1/models', undefined, signal);
		const list = res.ok && Array.isArray(res.data?.data) ? res.data.data : [];

		return list
			.filter((e) => ['loaded', 'loading'].includes(e.status?.value ?? 'loaded'))
			.map((e) => e.id);
	}

	/** POST /models/load or /models/unload for one model; true when the router took it. */
	private async modelCall(
		path: '/models/load' | '/models/unload',
		id: string,
		signal?: AbortSignal
	): Promise<boolean> {
		try {
			const res = await fetch(`${base}${path}`, {
				body: JSON.stringify({ model: id }),
				headers: getJsonHeaders(),
				method: 'POST',
				signal
			});

			return res.ok;
		} catch (e) {
			if (isAbort(e)) throw e;

			return false;
		}
	}

	private phaseText(): string {
		switch (this.phase) {
			case 'loading':
				return this.prep || 'Bench · preparing the prompts';
			case 'running':
				return `Bench running · step ${this.stepIndex} of ${this.stepCount}: ${this.stepName}${heldText(this.othersHeld)}`;
			case 'done':
				return this.overlapped
					? 'Bench done · other traffic shared the server, the run is marked · the bench chart and the speed card show it'
					: 'Bench done · the bench chart and the speed card show the new run';
			case 'stopped':
				return this.keptIncomplete
					? 'Bench stopped · what it measured is kept as an incomplete run'
					: 'Bench stopped · nothing measured yet';
			case 'error':
				return `Bench stopped · ${this.message}${this.keptIncomplete ? ' · what it measured is kept as an incomplete run' : ''}`;
			default:
				return '';
		}
	}

	/** A plain (not streamed) chat request with server timings. */
	private async plainChat(extra: Record<string, unknown>, signal: AbortSignal) {
		const stopWatch = this.watch();
		const t0 = performance.now();

		try {
			const res = await fetch(`${base}/v1/chat/completions`, {
				body: this.body(extra),
				headers: getJsonHeaders(),
				method: 'POST',
				signal: this.linkSignal(signal)
			});

			if (res.status === 401) throw new BenchError('the API key was refused');

			if (!res.ok) {
				return {
					error: `HTTP ${res.status} ${(await res.text()).slice(0, 300)}`,
					overlap: stopWatch(),
					text: '',
					timings: {} as Record<string, number>,
					wall_s: (performance.now() - t0) / 1000
				};
			}

			const json = (await res.json()) as {
				choices?: { message?: { content?: string } }[];
				timings?: Record<string, number>;
			};

			return {
				error: null,
				overlap: stopWatch(),
				text: json.choices?.[0]?.message?.content ?? '',
				timings: json.timings ?? {},
				wall_s: (performance.now() - t0) / 1000
			};
		} catch (e) {
			const overlap = stopWatch();

			if (e instanceof BenchError || signal.aborted) throw e;

			return {
				error: e instanceof Error ? `${e.name} ${e.message}`.slice(0, 200) : String(e),
				overlap,
				text: '',
				timings: {} as Record<string, number>,
				wall_s: (performance.now() - t0) / 1000
			};
		}
	}

	private plan(nCtx: number | null): StepPlan[] {
		const sessionFits = fits(nCtx, TOKENS_SESSION_MAX, 300);
		const steps: StepPlan[] = [];

		if (sessionFits) steps.push({ id: 'session1', name: 'agent session 1 of 2', requests: 14 });

		if (fits(nCtx, TOKENS_EDIT_MAX, EDIT_MAX_TOKENS))
			steps.push({ id: 'edits', name: 'code edits', requests: 3 });

		steps.push({ id: 'decode', name: 'decode answers', requests: PROSE.length + CODE.length });

		if (sessionFits) steps.push({ id: 'session2', name: 'agent session 2 of 2', requests: 14 });

		return steps;
	}

	/** Show what the run measured so far (the page treats it as the newest run). */
	private publish(): void {
		if (this.target) this.partial = this.snapshot({ inProgress: true });
	}

	private query(): Record<string, string | undefined> {
		return modelQuery(this.target?.modelId ?? null, this.target?.routerMode ?? false);
	}

	/** readme_bench.py chat(): temperature 1, server timings. */
	private async readmeCall(
		content: string,
		maxTokens: number,
		signal: AbortSignal
	): Promise<ReadmeCall> {
		const r = await this.plainChat(
			{ max_tokens: maxTokens, messages: [{ content, role: 'user' }], temperature: 1 },
			signal
		);

		this.done++;

		return {
			decode_tps: r.timings.predicted_per_second ?? null,
			draft_acc: r.timings.draft_n_accepted ?? 0,
			draft_n: r.timings.draft_n ?? 0,
			error: r.error,
			gen_tok: r.timings.predicted_n ?? null,
			overlap: r.overlap,
			prefill_tps: r.timings.prompt_per_second ?? null,
			prompt_tok: r.timings.prompt_n ?? null,
			wall_s: Math.round(r.wall_s * 10) / 10
		};
	}

	private async runEdits(fx: BenchFixtures, signal: AbortSignal): Promise<void> {
		for (const c of editCases(fx.edit)) {
			const r = await this.plainChat(
				{
					chat_template_kwargs: { enable_thinking: false },
					max_tokens: EDIT_MAX_TOKENS,
					messages: [{ content: `${c.task}\n\n${c.code}`, role: 'user' }],
					temperature: c.temp
				},
				signal
			);
			const lines = verbatimShare(c.code, r.text);

			this.parts.edits.push({
				case: c.name,
				decode_tps: r.timings.predicted_per_second ?? null,
				draft_acc: r.timings.draft_n_accepted ?? 0,
				draft_n: r.timings.draft_n ?? 0,
				error: r.error,
				gen: r.timings.predicted_n ?? null,
				overlap: r.overlap,
				prompt_n: r.timings.prompt_n ?? null,
				temp: c.temp,
				verbatim_lines: lines
			});
			this.done++;
			this.publish();
		}
	}

	private async runSession(fx: BenchFixtures, signal: AbortSignal): Promise<void> {
		const t0 = performance.now();

		this.parts.current = [];

		for (const req of buildSession(fx.agentic, sessionNonce())) {
			const r = await this.streamChat(req.messages, req.maxTokens, signal);

			this.parts.current.push({ ...r, conv: req.conv, kind: req.kind, step: req.name });
			this.done++;
			this.publish();
		}

		this.parts.sessions.push({
			session_s: (performance.now() - t0) / 1000,
			steps: this.parts.current
		});
		this.parts.current = [];
		this.publish();
	}

	private skippedSteps(nCtx: number | null): string[] {
		const out: string[] = [];

		if (!fits(nCtx, TOKENS_SESSION_MAX, 300)) out.push('agent sessions');

		if (!fits(nCtx, TOKENS_EDIT_MAX, EDIT_MAX_TOKENS)) out.push('code edits');

		return out;
	}

	/** The run as measured so far (a session still running counts with its finished requests). */
	private snapshot(flags: { incomplete?: boolean; inProgress?: boolean } = {}): BenchRun {
		const target = this.target as BenchTarget;
		const p = this.parts;
		const sessions = p.current.length
			? [...p.sessions, { session_s: 0, steps: p.current }]
			: p.sessions.slice();
		const parts = {
			code: readmeRow(p.code),
			edits: p.edits.slice(),
			prose: readmeRow(p.prose),
			sessions
		};
		const { labels, values } = runValues(parts);
		const overlapped =
			this.overlapped ||
			sessions.some((s) => s.steps.some((st) => st.overlap > 0)) ||
			p.edits.some((e) => e.overlap > 0) ||
			[...p.prose, ...p.code].some((c) => c.overlap > 0);

		return {
			...parts,
			...(flags.incomplete ? { incomplete: true } : {}),
			...(flags.inProgress ? { inProgress: true } : {}),
			build: target.build,
			labels,
			modelId: target.modelId,
			modelLabel: target.modelLabel,
			nCtx: target.nCtx,
			overlapped,
			skipped: this.skipped,
			t: Date.now(),
			v: 1,
			values
		};
	}

	/** agentic_bench.py chat(): a streaming request timed on the client. */
	private async streamChat(messages: ChatMessage[], maxTokens: number, signal: AbortSignal) {
		const stopWatch = this.watch();
		const t0 = performance.now();

		let tFirst: number | null = null;
		let tLast: number | null = null;
		let usage: Record<string, number> = {};
		let timings: Record<string, number> = {};
		let nDelta = 0;
		let error: string | null = null;

		try {
			const res = await fetch(`${base}/v1/chat/completions`, {
				body: this.body({
					max_tokens: maxTokens,
					messages,
					stream: true,
					stream_options: { include_usage: true },
					temperature: 1
				}),
				headers: getJsonHeaders(),
				method: 'POST',
				signal: this.linkSignal(signal)
			});

			if (res.status === 401) throw new BenchError('the API key was refused');

			if (!res.ok || !res.body) {
				error = `HTTP ${res.status} ${(await res.text()).slice(0, 300)}`;
			} else {
				const reader = res.body.getReader();
				const dec = new TextDecoder();

				let buf = '';

				outer: for (;;) {
					const { done, value } = await reader.read();

					if (done) break;

					buf += dec.decode(value, { stream: true });

					let nl: number;

					while ((nl = buf.indexOf('\n')) >= 0) {
						const line = buf.slice(0, nl).trim();

						buf = buf.slice(nl + 1);

						if (!line.startsWith('data:')) {
							if (line.startsWith('{') && line.includes('error')) error = line.slice(0, 300);

							continue;
						}

						const data = line.slice(5).trim();

						if (data === '[DONE]') break outer;

						let ch: {
							error?: unknown;
							usage?: Record<string, number>;
							timings?: Record<string, number>;
							choices?: { delta?: Record<string, unknown> }[];
						};

						try {
							ch = JSON.parse(data);
						} catch {
							continue;
						}

						if (ch.error) error = JSON.stringify(ch.error).slice(0, 300);

						if (ch.usage) usage = ch.usage;

						if (ch.timings) timings = ch.timings;

						const delta = ch.choices?.[0]?.delta ?? {};

						if (delta.content || delta.reasoning_content || delta.reasoning) {
							const now = performance.now();

							nDelta++;
							tFirst ??= now;
							tLast = now;
						}
					}
				}
			}
		} catch (e) {
			if (e instanceof BenchError || signal.aborted) {
				stopWatch();

				throw e;
			}

			error = e instanceof Error ? `${e.name} ${e.message}`.slice(0, 200) : String(e);
		}

		const tEnd = performance.now();
		const overlap = stopWatch();
		const nOut = usage.completion_tokens || timings.predicted_n || nDelta;

		return {
			completion_tokens: nOut ?? null,
			decode_tps:
				tFirst !== null && tLast !== null && tLast > tFirst && nOut > 1
					? (nOut - 1) / ((tLast - tFirst) / 1000)
					: null,
			draft_acc: timings.draft_n_accepted ?? null,
			draft_n: timings.draft_n ?? null,
			error,
			max_tokens: maxTokens,
			overlap,
			prompt_tokens: usage.prompt_tokens ?? null,
			srv_cache_n: timings.cache_n ?? null,
			srv_decode_tps: timings.predicted_per_second ?? null,
			srv_prefill_tps: timings.prompt_per_second ?? null,
			srv_prompt_n: timings.prompt_n ?? null,
			ttft_s: tFirst !== null ? (tFirst - t0) / 1000 : null,
			wall_s: (tEnd - t0) / 1000
		};
	}

	/**
	 * Unload every loaded model but the benched one (router mode), adding each to `unloaded` as the
	 * router takes it, and wait until they are gone (at most a minute). One it cannot unload (model
	 * management only from loopback, an error) stays loaded and marks the run.
	 */
	private async unloadOthers(signal: AbortSignal, unloaded: string[]): Promise<void> {
		const others = (await this.loadedModels(signal)).filter((id) => id !== this.target?.modelId);

		for (const id of others) {
			this.prep = `Bench · unloading ${id}`;

			if (await this.modelCall('/models/unload', id, signal)) unloaded.push(id);
			else this.overlapped = true;
		}

		const until = Date.now() + UNLOAD_WAIT_MS;

		while (unloaded.length > 0 && Date.now() < until) {
			const loaded = await this.loadedModels(signal);

			if (!unloaded.some((id) => loaded.includes(id))) break;

			await sleep(500, signal);
		}

		this.prep = '';
	}

	/** Polls /slots every second while a request runs; returns a stop function giving the count. */
	private watch(): () => number {
		let overlap = 0;
		let stopped = false;

		const tick = async () => {
			while (!stopped) {
				try {
					const slots = await this.getSlots();

					if (!stopped && slots) {
						const others = othersOnModel(slots);

						this.othersHeld = others;

						if (others > 0) overlap++;
					}
				} catch {
					// a refused key surfaces on the request itself
				}

				await new Promise((r) => setTimeout(r, 1000));
			}
		};

		void tick();

		return () => {
			stopped = true;

			return overlap;
		};
	}
}

export const benchRunner = new BenchRunner();
