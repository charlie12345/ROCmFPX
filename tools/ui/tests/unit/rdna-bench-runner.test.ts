// The bench runner end to end against a fake server (no network): it fills the run in as requests
// finish, saves the finished run, keeps a stopped run as incomplete; and the status page treats
// the run in progress as the newest run, glowing the figures it improves.
import { chartGroups } from '$lib/rdna/bench/metrics';
import { benchRunner, loadBenchRuns } from '$lib/rdna/bench/runner.svelte';
import type { BenchRun } from '$lib/rdna/bench/types';
import { StatusStore } from '$lib/rdna/status-store.svelte';
import { noteModelFile, writePart } from '$lib/rdna/storage';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const MODEL = 'qwen/qwen3.8-27b';
const store = new Map<string, string>();

let chats = 0;
let onChat: ((n: number) => void) | null = null;

const seenPartials: (BenchRun | null)[] = [];

function sse(body: Record<string, unknown>): Response {
	const n = Math.round(JSON.stringify(body.messages).length / 4);
	const max = Number(body.max_tokens) || 100;
	const lines = [
		`data: ${JSON.stringify({ choices: [{ delta: { content: 'a' } }] })}`,
		`data: ${JSON.stringify({ choices: [{ delta: { content: 'b' } }] })}`,
		`data: ${JSON.stringify({ choices: [], timings: { predicted_per_second: 60 }, usage: { completion_tokens: max, prompt_tokens: n } })}`,
		'data: [DONE]'
	];

	return new Response(lines.join('\n\n') + '\n\n', {
		headers: { 'Content-Type': 'text/event-stream' }
	});
}

function json(v: unknown): Response {
	return new Response(JSON.stringify(v), { headers: { 'Content-Type': 'application/json' } });
}

async function fakeFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
	const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;

	if (init?.signal?.aborted) throw new DOMException('stopped', 'AbortError');

	if (url.includes('agentic.json')) {
		return json({
			chunks: Array.from({ length: 9 }, (_, i) => `chunk ${i} `.repeat(40)),
			docs: 'docs',
			worker: 'worker'
		});
	}

	if (url.includes('edit.json')) return json({ code60000: 'int a;', code120000: 'int b;' });

	if (url.includes('/slots')) return json([{ id: 0, is_processing: false }]);

	if (url.includes('/requests')) return json({ requests: [] });

	if (url.includes('/v1/chat/completions')) {
		chats++;
		seenPartials.push(benchRunner.partial);
		onChat?.(chats);

		// a Stop during this request aborts it
		if (init?.signal?.aborted) throw new DOMException('stopped', 'AbortError');

		const body = JSON.parse(String(init?.body));

		expect(body.priority).toBe('xhigh');

		if (body.stream) return sse(body);

		return json({
			choices: [{ message: { content: 'int a;' } }],
			timings: {
				draft_n: 900,
				draft_n_accepted: 899,
				predicted_n: 1000,
				predicted_per_second: 300 + chats
			}
		});
	}

	return new Response('not found', { status: 404 });
}

const target = {
	build: 'b12781-707407de1',
	modelId: MODEL,
	modelLabel: 'Swift',
	nCtx: 98304,
	routerMode: true
};

beforeEach(() => {
	store.clear();
	chats = 0;
	onChat = null;
	seenPartials.length = 0;
	vi.stubGlobal('localStorage', {
		getItem: (k: string) => store.get(k) ?? null,
		removeItem: (k: string) => store.delete(k),
		setItem: (k: string, v: string) => store.set(k, v)
	});
	vi.stubGlobal('fetch', fakeFetch);
	// a server that names no model file: the plain keys
	noteModelFile(MODEL, null, null);
});

afterEach(() => {
	benchRunner.close();
	vi.unstubAllGlobals();
});

describe('bench runner', () => {
	it('fills the run in after every request and saves it when done', async () => {
		await benchRunner.start(target);

		expect(benchRunner.phase).toBe('done');
		// 14 + 3 + 18 + 14 requests, the first one before anything was measured
		expect(chats).toBe(49);
		expect(seenPartials[0]).toBeNull();
		// after the first request of the session the cold 4k figures are there already
		expect(seenPartials[1]?.inProgress).toBe(true);
		expect(seenPartials[1]?.values.prefill_4k).toBeGreaterThan(0);
		expect(seenPartials[1]?.values.session).toBeUndefined();
		// once the first session is complete its wall-clock counts
		expect(seenPartials[14]?.values.session).toBeGreaterThan(0);
		// the edits fill in the copy figure
		expect(seenPartials[15]?.values.edit).toBe(315);
		expect(seenPartials[16]?.values.edit).toBe(316);
		expect(benchRunner.partial).toBeNull();

		const runs = loadBenchRuns(MODEL);

		expect(runs).toHaveLength(1);
		expect(runs[0].incomplete).toBeUndefined();
		expect(runs[0].inProgress).toBeUndefined();
		expect(runs[0].values.edit).toBe(317);
	});

	it('keeps what a stopped run measured, marked incomplete', async () => {
		onChat = (n) => {
			if (n === 6) benchRunner.stop();
		};

		await benchRunner.start(target);

		expect(benchRunner.phase).toBe('stopped');
		expect(benchRunner.label).toContain('kept as an incomplete run');

		const runs = loadBenchRuns(MODEL);

		expect(runs).toHaveLength(1);
		expect(runs[0].incomplete).toBe(true);
		expect(runs[0].values.prefill_4k).toBeGreaterThan(0);
		expect(runs[0].values.session).toBeUndefined();
		expect(chartGroups(runs[0], null).map((g) => g.title)).toContain('PREFILL');
	});

	it('saves nothing when stopped before anything was measured', async () => {
		onChat = () => benchRunner.stop();

		await benchRunner.start(target);

		expect(benchRunner.label).toContain('nothing measured yet');
		expect(loadBenchRuns(MODEL)).toHaveLength(0);
	});
});

describe('the main tile follows the running bench', () => {
	function run(t: number, values: BenchRun['values'], flags: Partial<BenchRun> = {}): BenchRun {
		return {
			build: null,
			code: null,
			edits: [],
			labels: {},
			modelId: MODEL,
			modelLabel: 'Swift',
			nCtx: 98304,
			overlapped: false,
			prose: null,
			sessions: [],
			skipped: [],
			t,
			v: 1,
			values,
			...flags
		};
	}

	it('counts the run in progress as the newest run and glows what it improves', () => {
		writePart(MODEL, 'bench', [run(1, { decode_4k: 60, edit: 300, prefill_4k: 990, session: 97 })]);

		const page = new StatusStore();

		page.entries = [{ id: MODEL, meta: { n_ctx: 98304 }, status: { value: 'loaded' } }];
		page.select(MODEL);
		expect(page.card).toMatchObject({ decode: 60, empty: false, prefill: 990 });

		// the bench measured the cold 4k prompt faster: card and chart take it and glow
		benchRunner.modelId = MODEL;
		benchRunner.partial = run(2, { prefill_4k: 1002 }, { inProgress: true });
		page.observeRecords();

		expect(page.runsNewestFirst[0].inProgress).toBe(true);
		expect(page.card.prefill).toBe(1002);
		expect(page.card.decode).toBe(60);
		expect(page.glow.on('card:prefill', Date.now())).toBe(true);
		expect(page.glow.on('chart:prefill_4k', Date.now())).toBe(true);
		expect(page.glow.on('card:decode', Date.now())).toBe(false);

		// a shorter session is better: lower-is-better glows too
		benchRunner.partial = run(3, { prefill_4k: 1002, session: 95.5 }, { inProgress: true });
		page.observeRecords();
		expect(page.glow.on('chart:session', Date.now())).toBe(true);

		benchRunner.partial = null;
	});

	it('shows the empty card only until the first value of the first run', () => {
		const page = new StatusStore();

		page.entries = [{ id: MODEL, meta: { n_ctx: 98304 }, status: { value: 'loaded' } }];
		page.select(MODEL);
		expect(page.card.empty).toBe(true);

		benchRunner.modelId = MODEL;
		benchRunner.partial = run(5, {}, { inProgress: true });
		expect(page.card.empty).toBe(true);
		benchRunner.partial = run(6, { prefill_4k: 980 }, { inProgress: true });
		expect(page.card.empty).toBe(false);
		expect(page.card.prefill).toBe(980);
		benchRunner.partial = null;
	});
});
