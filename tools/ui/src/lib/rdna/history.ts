/**
 * RDNA-FAST status page: finished requests from GET /requests, merged into the history this browser
 * keeps, and everything derived from them: exact rates, peaks per depth bucket with records,
 * priority figures and the classifier's question types and answer times. Pure functions only.
 */

import {
	FAST_COPY_MIN_ACCEPTANCE,
	FAST_COPY_MIN_DRAFTS,
	LATENCY_SAMPLES_CAP,
	PEAK_MIN_DECODE_TOKENS,
	PEAK_MIN_PREFILL_TOKENS
} from './constants';
import { bucketIndex, type DepthBucket } from './depth';
import type {
	ClassifierMemory,
	ClassifierRecord,
	CompletionRecord,
	PeakMap,
	PriorityMemory,
	RequestRecord
} from './types';

function num(v: unknown): number | undefined {
	return typeof v === 'number' && Number.isFinite(v) ? v : undefined;
}

function str(v: unknown): string | undefined {
	return typeof v === 'string' ? v : undefined;
}

/**
 * One server record, checked and trimmed to the fields the page uses (never any text beyond the
 * classifier's answer label). Null for something that is not a finished request.
 */
export function normalizeRecord(raw: unknown): RequestRecord | null {
	if (!raw || typeof raw !== 'object') return null;

	const r = raw as Record<string, unknown>;
	const id = num(r.id);
	const tArrival = num(r.t_arrival_ms);
	const tEnd = num(r.t_end_ms);

	if (id === undefined || tArrival === undefined || tEnd === undefined) return null;

	if (r.kind === 'classifier') {
		const questions = Array.isArray(r.questions)
			? r.questions
					.slice(0, 64)
					.filter((q): q is Record<string, unknown> => !!q && typeof q === 'object')
					.map((q) => ({
						answer: str(q.answer)?.slice(0, 120),
						cached: typeof q.cached === 'boolean' ? q.cached : undefined,
						n_options: num(q.n_options),
						prob: num(q.prob),
						type: str(q.type)
					}))
			: undefined;
		const rec: ClassifierRecord = {
			answer: str(r.answer)?.slice(0, 120),
			cached: typeof r.cached === 'boolean' ? r.cached : undefined,
			id,
			kind: 'classifier',
			latency_ms: num(r.latency_ms),
			n_options: num(r.n_options),
			n_prompt: num(r.n_prompt),
			n_questions: num(r.n_questions),
			priority: str(r.priority),
			prob: num(r.prob),
			questions,
			status: str(r.status),
			t_arrival_ms: tArrival,
			t_end_ms: tEnd,
			type: str(r.type)
		};

		return rec;
	}

	const first = r.t_first_ms === null ? null : num(r.t_first_ms);
	const rec: CompletionRecord = {
		draft_accepted: num(r.draft_accepted),
		draft_n: num(r.draft_n),
		gen_ms: num(r.gen_ms),
		id,
		kind: 'completion',
		n_cache: num(r.n_cache),
		n_ctx: num(r.n_ctx),
		n_gen: num(r.n_gen),
		n_processed: num(r.n_processed),
		n_prompt: num(r.n_prompt),
		priority: str(r.priority),
		prompt_ms: num(r.prompt_ms),
		slot: num(r.slot),
		status: str(r.status),
		t_arrival_ms: tArrival,
		t_end_ms: tEnd,
		t_first_ms: first,
		t_start_ms: num(r.t_start_ms)
	};

	return rec;
}

/** Ids restart with the server; with the arrival time a key is unique across restarts. */
export function recordKey(r: RequestRecord): string {
	return `${r.kind}:${r.id}:${r.t_arrival_ms}`;
}

/**
 * Merge newly fetched records into the kept history: de-duplicated, newest first (by arrival),
 * capped. `added` holds the records the history did not have yet, oldest first.
 */
export function mergeHistory(
	history: readonly RequestRecord[],
	incoming: readonly RequestRecord[],
	cap: number
): { history: RequestRecord[]; added: RequestRecord[] } {
	const seen = new Set(history.map(recordKey));
	const added: RequestRecord[] = [];

	for (const r of incoming) {
		const key = recordKey(r);

		if (seen.has(key)) continue;

		seen.add(key);
		added.push(r);
	}

	if (added.length === 0) return { added, history: history.slice() };

	const merged = [...history, ...added].sort(
		(a, b) => b.t_arrival_ms - a.t_arrival_ms || b.id - a.id
	);

	added.sort((a, b) => a.t_end_ms - b.t_end_ms || a.id - b.id);

	return { added, history: merged.slice(0, cap) };
}

/** Prompt tokens per second of a finished request's prefill, null without one. */
export function prefillRate(r: CompletionRecord): number | null {
	if (!r.n_processed || !r.prompt_ms || r.n_processed <= 0 || r.prompt_ms <= 0) return null;

	return r.n_processed / (r.prompt_ms / 1000);
}

/**
 * Generated tokens per second as the server's predicted_per_second counts it: the first token
 * comes with the prefill, so the decode time covers the other n_gen - 1. Null below two tokens.
 */
export function decodeRate(r: CompletionRecord): number | null {
	if (!r.n_gen || !r.gen_ms || r.n_gen < 2 || r.gen_ms <= 0) return null;

	return (r.n_gen - 1) / (r.gen_ms / 1000);
}

/** Time to the first token as the client saw it, waiting in the queue included. */
export function ttftMs(r: CompletionRecord): number | null {
	if (r.t_first_ms === null || r.t_first_ms === undefined) return null;

	return Math.max(0, r.t_first_ms - r.t_arrival_ms);
}

/** Draft acceptance (0..1), null without drafts. */
export function draftAcceptance(r: CompletionRecord): number | null {
	if (!r.draft_n || r.draft_n <= 0) return null;

	return (r.draft_accepted ?? 0) / r.draft_n;
}

/**
 * A fast copy: the answer repeats text of the prompt, n-gram drafts carry nearly every token.
 * Its decode rate (often 250+ t/s) is shown in the Requests tile but is no decode peak.
 */
export function isFastCopy(r: CompletionRecord): boolean {
	const acc = draftAcceptance(r);

	return (
		acc !== null &&
		(r.draft_n ?? 0) >= FAST_COPY_MIN_DRAFTS &&
		(r.n_gen ?? 0) >= PEAK_MIN_DECODE_TOKENS &&
		acc >= FAST_COPY_MIN_ACCEPTANCE
	);
}

/** Depth a finished request ran at: its prompt (the context when decoding starts). */
export function recordDepth(r: CompletionRecord): number {
	if (r.n_prompt !== undefined) return r.n_prompt;

	if (r.n_ctx !== undefined) return Math.max(0, r.n_ctx - (r.n_gen ?? 0));

	return 0;
}

export interface PeakCandidate {
	/** The bucket's upper depth bound (peaks are keyed by it, so they survive a changed bucket list). */
	bucket: number;
	value: number;
	t: number;
}

/**
 * Decode and prefill rates of finished requests that may raise a peak (after a reset only, max
 * only): a request the polls never caught still counts by its exact rate. `decode` counts fast
 * copies too (the "Text copy boost" set), `decodeNoCopy` leaves them out.
 */
export function peakCandidates(
	records: readonly RequestRecord[],
	buckets: readonly DepthBucket[],
	decodeResetAt: number,
	prefillResetAt: number
): { decode: PeakCandidate[]; decodeNoCopy: PeakCandidate[]; prefill: PeakCandidate[] } {
	const decode: PeakCandidate[] = [];
	const decodeNoCopy: PeakCandidate[] = [];
	const prefill: PeakCandidate[] = [];

	for (const r of records) {
		if (r.kind !== 'completion' || r.status === 'error') continue;

		const b = bucketIndex(recordDepth(r), buckets);

		if (b < 0) continue;

		const hi = buckets[b].hi;
		const d = decodeRate(r);

		if (d !== null && r.t_end_ms > decodeResetAt && (r.n_gen ?? 0) >= PEAK_MIN_DECODE_TOKENS) {
			decode.push({ bucket: hi, t: r.t_end_ms, value: d });

			if (!isFastCopy(r)) decodeNoCopy.push({ bucket: hi, t: r.t_end_ms, value: d });
		}

		const p = prefillRate(r);

		if (
			p !== null &&
			r.t_end_ms > prefillResetAt &&
			(r.n_processed ?? 0) >= PEAK_MIN_PREFILL_TOKENS
		) {
			prefill.push({ bucket: hi, t: r.t_end_ms, value: p });
		}
	}

	return { decode, decodeNoCopy, prefill };
}

/** Raise the peaks with new candidates. `records` lists the buckets whose peak went up. */
export function updatePeaks(
	peaks: PeakMap,
	candidates: readonly PeakCandidate[]
): { peaks: PeakMap; records: number[] } {
	const out: PeakMap = { ...peaks };
	const records = new Set<number>();

	for (const c of candidates) {
		const old = out[c.bucket];

		if (!old || c.value > old.v) {
			out[c.bucket] = { t: c.t, v: c.value };
			records.add(c.bucket);
		}
	}

	return { peaks: out, records: Array.from(records).sort((a, b) => a - b) };
}

/** Add finished completion requests to the priority figures (requests and TTFT since reset). */
export function addToPriority(
	mem: PriorityMemory,
	records: readonly RequestRecord[]
): PriorityMemory {
	const byPriority = { ...mem.byPriority };

	let changed = false;

	for (const r of records) {
		if (r.kind !== 'completion' || r.t_end_ms <= mem.resetAt) continue;

		const key = r.priority || 'default';
		const cur = byPriority[key] ?? { n: 0, ttftN: 0, ttftSum: 0 };
		const t = ttftMs(r);

		byPriority[key] = {
			n: cur.n + 1,
			ttftN: cur.ttftN + (t !== null ? 1 : 0),
			ttftSum: cur.ttftSum + (t ?? 0)
		};
		changed = true;
	}

	return changed ? { ...mem, byPriority } : mem;
}

/** Question type key of a classifier answer: `noul`, `choice:<N>` or `score`; null without a type. */
export function questionTypeKey(r: Pick<ClassifierRecord, 'type' | 'n_options'>): string | null {
	const type = (r.type ?? '').trim().toLowerCase();

	if (!type) return null;

	if (type === 'choice') return `choice:${r.n_options ?? 0}`;

	return type;
}

/** The question type keys of a request: one per question when the server lists them. */
export function questionTypeKeys(r: ClassifierRecord): string[] {
	const list = r.questions && r.questions.length > 0 ? r.questions : [r];

	return list.map(questionTypeKey).filter((k): k is string => k !== null);
}

export function questionTypeLabel(key: string): string {
	if (key === 'noul') return 'yes / no';

	if (key === 'score') return 'score';

	if (key.startsWith('choice:')) return `choice of ${key.slice('choice:'.length)}`;

	return key;
}

function questionTypeRank(key: string): number {
	if (key === 'noul') return -1;

	if (key === 'score') return 1e9;

	if (key.startsWith('choice:')) return Number(key.slice('choice:'.length)) || 0;

	return 1e8;
}

/** Rows of the question types tile: yes / no first, choices by option count, score last. */
export function questionTypeRows(
	types: Record<string, number>
): { key: string; label: string; count: number }[] {
	return Object.keys(types)
		.sort((a, b) => questionTypeRank(a) - questionTypeRank(b) || a.localeCompare(b))
		.map((key) => ({ count: types[key], key, label: questionTypeLabel(key) }));
}

/** Prompt tokens per second of a classifier answer computed by the model (not the cache). */
export function classifierPromptRate(r: ClassifierRecord): number | null {
	if (r.cached || !r.n_prompt || !r.latency_ms || r.latency_ms <= 0) return null;

	return r.n_prompt / (r.latency_ms / 1000);
}

/** Add classifier answers to the speed and question type figures (each since its own reset). */
export function addToClassifier(
	mem: ClassifierMemory,
	records: readonly RequestRecord[]
): ClassifierMemory {
	let out = mem;

	for (const r of records) {
		if (r.kind !== 'classifier' || r.status === 'error') continue;

		if (r.t_end_ms > out.typesResetAt) {
			const keys = questionTypeKeys(r);

			if (keys.length > 0) {
				const types = { ...out.types };

				for (const key of keys) types[key] = (types[key] ?? 0) + 1;

				out = { ...out, types };
			}
		}

		if (r.t_end_ms > out.speedResetAt) {
			const rate = classifierPromptRate(r);
			const latencies =
				r.latency_ms !== undefined
					? [...out.latencies, r.latency_ms].slice(-LATENCY_SAMPLES_CAP)
					: out.latencies;
			const own = !r.cached && r.latency_ms !== undefined && r.latency_ms > 0 ? r.latency_ms : null;

			out = {
				...out,
				fastest: own !== null ? Math.min(out.fastest ?? own, own) : out.fastest,
				firstT: out.firstT === null ? r.t_end_ms : Math.min(out.firstT, r.t_end_ms),
				latencies,
				n: out.n + 1,
				promptPeak: rate !== null ? Math.max(out.promptPeak ?? 0, rate) : out.promptPeak
			};
		}
	}

	return out;
}

/** Nearest-rank percentile (0..100) of a list, null when empty. */
export function percentile(values: readonly number[], p: number): number | null {
	if (values.length === 0) return null;

	const sorted = [...values].sort((a, b) => a - b);
	const rank = Math.ceil((p / 100) * sorted.length);

	return sorted[Math.min(sorted.length - 1, Math.max(0, rank - 1))];
}

/** The middle value (mean of the two middle values for an even count), null when empty. */
export function median(values: readonly number[]): number | null {
	if (values.length === 0) return null;

	const sorted = [...values].sort((a, b) => a - b);
	const mid = Math.floor(sorted.length / 2);

	return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/** Answers per minute since the first answer after the reset (at least one minute of span). */
export function answersPerMinute(mem: ClassifierMemory, now: number): number | null {
	if (mem.n === 0 || mem.firstT === null) return mem.n === 0 ? 0 : null;

	const minutes = Math.max(1, (now - Math.max(mem.firstT, mem.speedResetAt)) / 60000);

	return mem.n / minutes;
}
