/**
 * RDNA-FAST bench: the figures of a run, computed the way the owner's scripts compute them.
 *   fair_table.py   per run, the better of its two sessions: session = the shorter wall-clock, decode
 *                   at 4k / 17k / 41k = the higher client decode t/s of that step, prefill = new prompt
 *                   tokens (this orchestrator request's prompt minus the previous one's, the whole prompt
 *                   for the cold 4k) over the client's time to the first token; edit = the best edit
 *   readme_bench.py median and peak of nine decodes
 * The speed card shows the best of the last two runs; the bench chart this run against the one before.
 */

import { PICK_4K, PICK_17K, PICK_41K } from './scenario';
import type {
	BenchLabels,
	BenchRun,
	BenchValueKey,
	BenchValues,
	EditRow,
	ReadmeCall,
	ReadmeRow,
	SessionResult,
	SessionStep
} from './types';

/** fair_table.py prefill(): new prompt tokens over the client's time to the first token. */
export function sessionPrefill(steps: readonly SessionStep[], name: string): number | null {
	let prev: SessionStep | null = null;

	for (const s of steps) {
		if (s.conv !== 'A') continue;

		if (s.step === name) {
			if (s.prompt_tokens === null || !s.ttft_s || s.ttft_s <= 0 || s.error) return null;

			const before = prev && name !== PICK_4K ? (prev.prompt_tokens ?? 0) : 0;

			return (s.prompt_tokens - before) / s.ttft_s;
		}

		prev = s;
	}

	return null;
}

/** New prompt tokens of a measuring point (for the labels). */
export function sessionNewTokens(steps: readonly SessionStep[], name: string): number | null {
	let prev: SessionStep | null = null;

	for (const s of steps) {
		if (s.conv !== 'A') continue;

		if (s.step === name) {
			if (s.prompt_tokens === null) return null;

			return s.prompt_tokens - (prev && name !== PICK_4K ? (prev.prompt_tokens ?? 0) : 0);
		}

		prev = s;
	}

	return null;
}

function maxOf(values: (number | null | undefined)[]): number | undefined {
	const v = values.filter((x): x is number => typeof x === 'number' && Number.isFinite(x));

	return v.length > 0 ? Math.max(...v) : undefined;
}

function minOf(values: (number | null | undefined)[]): number | undefined {
	const v = values.filter((x): x is number => typeof x === 'number' && Number.isFinite(x));

	return v.length > 0 ? Math.min(...v) : undefined;
}

function stepOf(session: SessionResult, name: string): SessionStep | undefined {
	return session.steps.find((s) => s.step === name);
}

/** A session that ran all its requests without an error counts for the wall-clock. */
export function sessionComplete(session: SessionResult): boolean {
	return session.steps.length === 14 && session.steps.every((s) => !s.error);
}

/** readme_bench.py row(): median, peak and min of the decode rates (rounded to 0.1 first). */
export function readmeRow(calls: readonly ReadmeCall[]): ReadmeRow | null {
	const ok = calls.filter((c) => !c.error && typeof c.decode_tps === 'number');

	if (ok.length === 0) return null;

	const tps = ok.map((c) => Math.round((c.decode_tps as number) * 10) / 10).sort((a, b) => a - b);
	const mid = Math.floor(tps.length / 2);
	const med = tps.length % 2 ? tps[mid] : (tps[mid - 1] + tps[mid]) / 2;
	const acc = ok.reduce((s, c) => s + c.draft_acc, 0);
	const drn = ok.reduce((s, c) => s + c.draft_n, 0);

	return {
		draft_acceptance: drn ? Math.round((acc / drn) * 1000) / 1000 : null,
		median: Math.round(med * 10) / 10,
		min: tps[0],
		n: ok.length,
		peak: tps[tps.length - 1]
	};
}

export interface RunParts {
	sessions: SessionResult[];
	edits: EditRow[];
	prose: ReadmeRow | null;
	code: ReadmeRow | null;
}

/** The figures of one run. */
export function runValues(parts: RunParts): { values: BenchValues; labels: BenchLabels } {
	const values: BenchValues = {};
	const labels: BenchLabels = {};
	const sessions = parts.sessions;
	const set = (k: BenchValueKey, v: number | undefined) => {
		if (v !== undefined) values[k] = v;
	};

	set('session', minOf(sessions.filter(sessionComplete).map((s) => s.session_s)));

	for (const [key, name] of [
		['4k', PICK_4K],
		['17k', PICK_17K],
		['41k', PICK_41K]
	] as const) {
		set(
			`decode_${key}`,
			maxOf(
				sessions.map((s) => {
					const st = stepOf(s, name);

					return st && !st.error ? st.decode_tps : null;
				})
			)
		);
		set(`prefill_${key}`, maxOf(sessions.map((s) => sessionPrefill(s.steps, name))));

		const ref = sessions.map((s) => stepOf(s, name)).find((st) => st && st.prompt_tokens !== null);

		if (ref?.prompt_tokens) labels[`tok_${key}`] = ref.prompt_tokens;

		if (key !== '4k') {
			const n = sessions.map((s) => sessionNewTokens(s.steps, name)).find((x) => x !== null);

			if (n !== null && n !== undefined) labels[`new_${key}`] = n;
		}
	}

	set('edit', maxOf(parts.edits.filter((e) => !e.error).map((e) => e.decode_tps)));

	if (parts.prose) set('prose', parts.prose.median);

	if (parts.code) set('code', parts.code.median);

	return { labels, values };
}

const LOWER_IS_BETTER: ReadonlySet<BenchValueKey> = new Set(['session']);

/** The figures the bench chart shows (each can glow when a run improves it). */
export const BENCH_CHART_KEYS: readonly BenchValueKey[] = [
	'session',
	'decode_4k',
	'decode_17k',
	'decode_41k',
	'prefill_4k',
	'prefill_17k',
	'prefill_41k',
	'edit',
	'prose',
	'code'
];

/** The best value of each figure over the newest two runs (the speed card). */
export function bestOfLastTwo(runs: readonly BenchRun[]): {
	values: BenchValues;
	runs: BenchRun[];
} {
	const last = [...runs].sort((a, b) => b.t - a.t).slice(0, 2);
	const values: BenchValues = {};

	for (const run of last) {
		for (const [k, v] of Object.entries(run.values) as [BenchValueKey, number][]) {
			if (typeof v !== 'number') continue;

			const cur = values[k];

			if (cur === undefined) values[k] = v;
			else values[k] = LOWER_IS_BETTER.has(k) ? Math.min(cur, v) : Math.max(cur, v);
		}
	}

	return { runs: last, values };
}

/** Newest run first. */
export function sortedRuns(runs: readonly BenchRun[]): BenchRun[] {
	return [...runs].sort((a, b) => b.t - a.t);
}

export interface ChartRow {
	/** The figure (for the record glow). */
	key: BenchValueKey;
	label: string;
	now: number | null;
	prev: number | null;
}

export interface ChartGroup {
	title: string;
	sub: string;
	unit: string;
	rows: ChartRow[];
}

function k1(n: number | undefined): string | null {
	return n ? `${(n / 1000).toFixed(1)}k` : null;
}

function kr(n: number | undefined, fallback: string): string {
	return n ? `${Math.round(n / 1000)}k` : fallback;
}

/** The bench chart's groups: this run against the previous one. Rows without any value are left out. */
export function chartGroups(now: BenchRun | null, prev: BenchRun | null): ChartGroup[] {
	const v = (run: BenchRun | null, key: BenchValueKey) => run?.values[key] ?? null;
	const lab = now?.labels ?? prev?.labels ?? {};
	const groups: ChartGroup[] = [
		{
			rows: [
				{
					key: 'session',
					label: 'two-agent session',
					now: v(now, 'session'),
					prev: v(prev, 'session')
				}
			],
			sub: 'wall-clock, two agents, 14 requests, lower is better',
			title: 'AGENTIC SESSION',
			unit: 's'
		},
		{
			rows: (['4k', '17k', '41k'] as const).map((key) => ({
				key: `decode_${key}` as const,
				label: `at ${kr(lab[`tok_${key}`], key)} depth`,
				now: v(now, `decode_${key}`),
				prev: v(prev, `decode_${key}`)
			})),
			sub: 'generated tokens per second',
			title: 'DECODE',
			unit: 't/s'
		},
		{
			rows: [
				{
					key: 'prefill_4k' as const,
					label: `cold ${kr(lab.tok_4k, '4k')} prompt`,
					now: v(now, 'prefill_4k'),
					prev: v(prev, 'prefill_4k')
				},
				...(['17k', '41k'] as const).map((key) => {
					const n = k1(lab[`new_${key}`]);

					return {
						key: `prefill_${key}` as const,
						label: `${n ? `${n} new` : 'new tokens'} at ${kr(lab[`tok_${key}`], key)} depth`,
						now: v(now, `prefill_${key}`),
						prev: v(prev, `prefill_${key}`)
					};
				})
			],
			sub: 'prompt tokens per second',
			title: 'PREFILL',
			unit: 't/s'
		},
		{
			rows: [
				{
					key: 'edit',
					label: 'one small change to a file',
					now: v(now, 'edit'),
					prev: v(prev, 'edit')
				}
			],
			sub: 'decode when the answer copies the prompt',
			title: 'TEXT COPY BOOST',
			unit: 't/s'
		},
		{
			rows: [
				{ key: 'prose', label: 'prose', now: v(now, 'prose'), prev: v(prev, 'prose') },
				{ key: 'code', label: 'code', now: v(now, 'code'), prev: v(prev, 'code') }
			],
			sub: 'median of nine 450-token answers',
			title: 'DECODE ANSWERS',
			unit: 't/s'
		}
	];

	return groups
		.map((g) => ({ ...g, rows: g.rows.filter((r) => r.now !== null || r.prev !== null) }))
		.filter((g) => g.rows.length > 0);
}
