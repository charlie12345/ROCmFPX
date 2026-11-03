import d1103 from './fixtures/rdna-d1103-sessions.json';
import {
	bestOfLastTwo,
	chartGroups,
	readmeRow,
	runValues,
	sessionComplete,
	sessionNewTokens,
	sessionPrefill
} from '$lib/rdna/bench/metrics';
import {
	buildSession,
	editCases,
	fits,
	PICK_4K,
	PICK_17K,
	PICK_41K,
	sessionNonce,
	verbatimShare
} from '$lib/rdna/bench/scenario';
import type { BenchRun, EditRow, ReadmeCall, SessionResult } from '$lib/rdna/bench/types';
import { describe, expect, it } from 'vitest';

// the owner's two sessions of 2026-11-03 (benchmarks/stock-vs-prod/d1103-chart-session1/2.json)
const SESSIONS = (d1103 as unknown as Record<string, SessionResult[]>)['d1103-chart'];

function edit(decode: number, error: string | null = null): EditRow {
	return {
		case: 'rename',
		decode_tps: decode,
		draft_acc: 962,
		draft_n: 963,
		error,
		gen: 1000,
		overlap: 0,
		prompt_n: 2285,
		temp: 0,
		verbatim_lines: 0.98
	};
}

function call(decode: number, extra: Partial<ReadmeCall> = {}): ReadmeCall {
	return {
		decode_tps: decode,
		draft_acc: 6,
		draft_n: 10,
		error: null,
		gen_tok: 450,
		overlap: 0,
		prefill_tps: null,
		prompt_tok: 40,
		wall_s: 8,
		...extra
	};
}

function run(t: number, values: BenchRun['values'], labels: BenchRun['labels'] = {}): BenchRun {
	return {
		build: 'b12781-707407de1',
		code: null,
		edits: [],
		labels,
		modelId: 'qwen/qwen3.8-27b',
		modelLabel: 'Swift-1.5-Qwen3.8-27B IQ4_XS',
		nCtx: 98304,
		overlapped: false,
		prose: null,
		sessions: [],
		skipped: [],
		t,
		v: 1,
		values
	};
}

describe('fair_table.py port', () => {
	it('reproduces fair_table.py on the owner’s d1103-chart run', () => {
		const { labels, values } = runValues({
			code: null,
			edits: [edit(303.3), edit(274.7), edit(310.9)],
			prose: null,
			sessions: SESSIONS
		});

		// python fair_table.py d1103-chart
		expect(values.session).toBeCloseTo(97.32799999997951, 9);
		expect(values.decode_4k).toBeCloseTo(60.94578067578525, 9);
		expect(values.decode_17k).toBeCloseTo(60.36745406849119, 9);
		expect(values.decode_41k).toBeCloseTo(61.145194273853946, 9);
		expect(values.prefill_4k).toBeCloseTo(1002.2410358412338, 9);
		expect(values.prefill_17k).toBeCloseTo(856.97352824387, 9);
		expect(values.prefill_41k).toBeCloseTo(677.8136139803661, 9);
		expect(values.edit).toBe(310.9);
		expect(labels).toEqual({
			new_17k: 4338,
			new_41k: 3993,
			tok_4k: 4025,
			tok_17k: 17039,
			tok_41k: 41053
		});
	});

	it('counts new prompt tokens from the previous orchestrator request, the cold 4k whole', () => {
		const steps = SESSIONS[0].steps;

		expect(sessionNewTokens(steps, PICK_4K)).toBe(4025);
		expect(sessionNewTokens(steps, PICK_17K)).toBe(17039 - 12701);
		expect(sessionNewTokens(steps, PICK_41K)).toBe(41053 - 37060);
		expect(sessionPrefill(steps, 'nope')).toBeNull();
	});

	it('leaves failed steps and incomplete sessions out', () => {
		const broken: SessionResult = {
			session_s: 50,
			steps: SESSIONS[0].steps.map((s) =>
				s.step === PICK_4K ? { ...s, error: 'HTTP 500', ttft_s: null } : s
			)
		};

		expect(sessionComplete(SESSIONS[0])).toBe(true);
		expect(sessionComplete(broken)).toBe(false);
		expect(sessionPrefill(broken.steps, PICK_4K)).toBeNull();

		const v = runValues({
			code: null,
			edits: [edit(300, 'HTTP 400')],
			prose: null,
			sessions: [broken]
		}).values;

		expect(v.session).toBeUndefined();
		expect(v.prefill_4k).toBeUndefined();
		expect(v.edit).toBeUndefined();
		expect(v.prefill_17k).toBeGreaterThan(0);
	});
});

describe('readme_bench.py port', () => {
	it('summarises nine decodes like row()', () => {
		const r = readmeRow([54.5, 58.84, 51.8, 53.3, 55, 56.1, 52.2, 57.7, 54.04].map((v) => call(v)));

		expect(r).toEqual({ draft_acceptance: 0.6, median: 54.5, min: 51.8, n: 9, peak: 58.8 });
		expect(readmeRow([call(50, { error: 'x' })])).toBeNull();
	});

	it('takes decode answers as their medians; old runs’ cold-prompt figures are ignored', () => {
		const v = runValues({
			code: readmeRow([call(82.5)]),
			edits: [],
			prose: readmeRow([call(54.5)]),
			sessions: []
		});

		expect(v.values).toEqual({ code: 82.5, prose: 54.5 });

		// a run saved before the cold prompts were dropped still reads, without a cold-prompt group
		const old = {
			...run(1, { cold_1: 955.2, decode_4k: 60 } as BenchRun['values']),
			cold: [{ name: '17k' }]
		} as BenchRun;

		expect(chartGroups(old, null).map((g) => g.title)).toEqual(['DECODE']);
		expect(bestOfLastTwo([old]).values.decode_4k).toBe(60);
	});
});

describe('speed card and chart', () => {
	it('shows the best of the last two runs (the shorter session)', () => {
		const runs = [
			run(1, { decode_4k: 70, prefill_4k: 1100, session: 90 }),
			run(3, { decode_4k: 60.9, edit: 310.9, prefill_4k: 1002.2, session: 97.3 }),
			run(2, { decode_4k: 65.8, edit: 306.4, prefill_4k: 990.6, session: 96.7 })
		];
		const best = bestOfLastTwo(runs);

		expect(best.runs.map((r) => r.t)).toEqual([3, 2]);
		expect(best.values).toEqual({
			decode_4k: 65.8,
			edit: 310.9,
			prefill_4k: 1002.2,
			session: 96.7
		});
		expect(bestOfLastTwo([]).values).toEqual({});
	});

	it('labels the chart rows from the run’s token counts and drops empty rows', () => {
		const now = run(
			2,
			{
				decode_4k: 65.8,
				decode_17k: 64,
				edit: 311,
				prefill_4k: 1002,
				prefill_17k: 857,
				session: 96.7
			},
			{ new_17k: 4338, tok_4k: 4025, tok_17k: 17039 }
		);
		const groups = chartGroups(now, run(1, { session: 95.9 }));

		expect(groups.map((g) => g.title)).toEqual([
			'AGENTIC SESSION',
			'DECODE',
			'PREFILL',
			'TEXT COPY BOOST'
		]);
		expect(groups[1].rows.map((r) => r.label)).toEqual(['at 4k depth', 'at 17k depth']);
		expect(groups[2].rows.map((r) => r.label)).toEqual(['cold 4k prompt', '4.3k new at 17k depth']);
		expect(groups[0].rows[0]).toEqual({
			key: 'session',
			label: 'two-agent session',
			now: 96.7,
			prev: 95.9
		});
		expect(chartGroups(null, null)).toEqual([]);
	});
});

describe('bench prompts', () => {
	const fx = {
		chunks: Array.from({ length: 9 }, (_, i) => `chunk${i}`),
		docs: 'DOCS',
		worker: 'WORKER'
	};

	it('builds the 14 requests of agentic_bench.py in order', () => {
		const reqs = buildSession(fx, 'abc123');

		expect(reqs.map((r) => r.name)).toEqual([
			PICK_4K,
			'A2 + file',
			'A3 + file',
			PICK_17K,
			'B1 worker, cold (10k)',
			'A6 back at 17k',
			'A7 + file',
			'A8 + file',
			'A9 + file',
			'A10 + file',
			'A11 + file',
			PICK_41K,
			'B2 worker again (10k)',
			'A14 back at 41k'
		]);
		expect(reqs.map((r) => r.maxTokens)).toEqual([
			300, 100, 100, 300, 200, 150, 100, 100, 100, 100, 100, 300, 200, 150
		]);
		expect(reqs.map((r) => r.messages.length)).toEqual([
			2, 4, 6, 8, 2, 10, 12, 14, 16, 18, 20, 22, 2, 24
		]);
		expect(reqs[0].messages[0].content).toBe(
			'Session id: abc123\nYou are the orchestrator agent of a software team. You plan work, read files through tools and review code. The reference documentation of the server you maintain follows.\n\nDOCS'
		);
		expect(reqs[1].messages[3].content).toBe(
			'[tool result: read_file part 1 of tools/server/server-context.cpp]\nchunk0\n\nReview this part: what does it do and what would you change? Answer with a detailed written analysis of at least 600 words, in prose.'
		);
		expect(reqs[12].messages[0]).toBe(reqs[4].messages[0]);
		expect(
			reqs[4].messages[0].content.startsWith('Session id: abc123-w\nYou are a worker agent.')
		).toBe(true);
	});

	it('builds the edit cases and scores verbatim lines like edit_gate.py', () => {
		const cases = editCases({ code60000: 'A', code120000: 'B' });

		expect(cases.map((c) => [c.name, c.temp, c.code])).toEqual([
			['rename', 0, 'A'],
			['comment', 0, 'B'],
			['rename1', 1, 'A']
		]);
		expect(verbatimShare('int a;\nint b;\n', 'int a;\nint c;\n\n int b; ')).toBe(0.67);
		expect(verbatimShare('x', '')).toBe(0);
	});

	it('starts every session with a fresh id so a prompt cache cannot serve it', () => {
		expect(sessionNonce()).toMatch(/^[0-9a-f]{12}$/);
		expect(sessionNonce()).not.toBe(sessionNonce());
	});

	it('skips what the context cannot hold', () => {
		expect(fits(98304, 41107, 300)).toBe(true);
		expect(fits(32768, 41107, 300)).toBe(false);
		expect(fits(65536, 41107, 300)).toBe(true);
		expect(fits(null, 1e9, 1)).toBe(true);
	});
});
