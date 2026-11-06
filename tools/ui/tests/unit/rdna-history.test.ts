import { addToCache, avgRestoreMs, cacheCountersWentDown, cacheDelta } from '$lib/rdna/cache';
import {
	bucketIndex,
	DEPTH_OPEN_HI,
	depthBuckets,
	migratePeakMap,
	niceScale
} from '$lib/rdna/depth';
import {
	addToClassifier,
	addToPriority,
	answersPerMinute,
	decodeRate,
	isFastCopy,
	median,
	mergeHistory,
	normalizeRecord,
	peakCandidates,
	percentile,
	prefillRate,
	questionTypeKey,
	questionTypeKeys,
	questionTypeLabel,
	questionTypeRows,
	recordKey,
	ttftMs,
	updatePeaks
} from '$lib/rdna/history';
import {
	emptyCache,
	emptyClassifier,
	emptyPriority,
	loadPeaks,
	noteModelFile,
	writePart
} from '$lib/rdna/storage';
import type { ClassifierRecord, CompletionRecord, RequestRecord } from '$lib/rdna/types';
import { describe, expect, it, vi } from 'vitest';

function completion(over: Partial<CompletionRecord> = {}): CompletionRecord {
	return {
		draft_accepted: 0,
		draft_n: 0,
		gen_ms: 1000,
		id: 1,
		kind: 'completion',
		n_cache: 0,
		n_ctx: 4400,
		n_gen: 101,
		n_processed: 4000,
		n_prompt: 4300,
		priority: 'default',
		prompt_ms: 4000,
		slot: 0,
		status: 'done',
		t_arrival_ms: 1000,
		t_end_ms: 7000,
		t_first_ms: 5500,
		t_start_ms: 1500,
		...over
	};
}

function classifier(over: Partial<ClassifierRecord> = {}): ClassifierRecord {
	return {
		answer: 'yes',
		cached: false,
		id: 1,
		kind: 'classifier',
		latency_ms: 268,
		n_options: 2,
		n_prompt: 106,
		priority: 'default',
		prob: 0.69,
		status: 'done',
		t_arrival_ms: 1000,
		t_end_ms: 1268,
		type: 'noul',
		...over
	};
}

describe('depth buckets', () => {
	it('has three rows: up to 8k, 8k to 24k, 24k+', () => {
		expect(depthBuckets(98304).map((b) => b.label)).toEqual(['up to 8k', '8k to 24k', '24k+']);
		expect(depthBuckets(262144).map((b) => b.label)).toEqual(['up to 8k', '8k to 24k', '24k+']);
		expect(depthBuckets(null)).toHaveLength(3);
		// a context that ends before a bucket starts does not show it
		expect(depthBuckets(16384).map((b) => b.label)).toEqual(['up to 8k', '8k to 24k']);
		expect(depthBuckets(4096).map((b) => b.label)).toEqual(['up to 8k']);
		expect(depthBuckets(98304)[2]).toEqual({ hi: DEPTH_OPEN_HI, label: '24k+', lo: 24576 });
	});

	it('puts a depth into its bucket, everything from 24k on into 24k+', () => {
		const b = depthBuckets(98304);

		expect(bucketIndex(0, b)).toBe(0);
		expect(bucketIndex(8191, b)).toBe(0);
		expect(bucketIndex(8192, b)).toBe(1);
		expect(bucketIndex(17039, b)).toBe(1);
		expect(bucketIndex(24576, b)).toBe(2);
		expect(bucketIndex(41053, b)).toBe(2);
		expect(bucketIndex(69086, b)).toBe(2);
		expect(bucketIndex(5_000_000, b)).toBe(2);
		expect(bucketIndex(5, [])).toBe(-1);
	});

	it('folds stored 24k-48k and 48k+ peaks into 24k+ by their maximum', () => {
		expect(
			migratePeakMap({
				8192: { t: 1, v: 90.6 },
				24576: { t: 2, v: 71.4 },
				49152: { t: 3, v: 63.9 },
				98304: { t: 4, v: 54.2 },
				131072: { t: 5, v: 66.1 }
			})
		).toEqual({
			8192: { t: 1, v: 90.6 },
			24576: { t: 2, v: 71.4 },
			[DEPTH_OPEN_HI]: { t: 5, v: 66.1 }
		});
		// already migrated stores stay as they are
		const today = { 8192: { t: 1, v: 1 }, [DEPTH_OPEN_HI]: { t: 2, v: 2 } };

		expect(migratePeakMap(today)).toEqual(today);
	});

	it('migrates when the peaks are loaded', () => {
		const store = new Map<string, string>();

		vi.stubGlobal('localStorage', {
			getItem: (k: string) => store.get(k) ?? null,
			removeItem: (k: string) => store.delete(k),
			setItem: (k: string, v: string) => store.set(k, v)
		});
		noteModelFile('m', null, null);
		writePart('m', 'peaks', {
			decode: { 49152: { t: 3, v: 63.9 }, 98304: { t: 4, v: 54.2 } },
			decodeResetAt: 0,
			prefill: { 98304: { t: 4, v: 640 } },
			prefillResetAt: 0
		});

		const p = loadPeaks('m');

		expect(p.decode).toEqual({ [DEPTH_OPEN_HI]: { t: 3, v: 63.9 } });
		expect(p.prefill).toEqual({ [DEPTH_OPEN_HI]: { t: 4, v: 640 } });
		vi.unstubAllGlobals();
	});

	it('rounds bar scales up a little', () => {
		expect(niceScale(90.6)).toBe(100);
		expect(niceScale(1002)).toBe(1200);
		expect(niceScale(0)).toBe(1);
	});
});

describe('records', () => {
	it('keeps only known fields and never text beyond the answer label', () => {
		const raw = {
			...completion(),
			prompt: 'secret prompt',
			t_first_ms: null
		};
		const r = normalizeRecord(raw) as CompletionRecord;

		expect(r.kind).toBe('completion');
		expect(r.t_first_ms).toBeNull();
		expect('prompt' in r).toBe(false);
		expect(normalizeRecord({ id: 1 })).toBeNull();
		expect(normalizeRecord('x')).toBeNull();
	});

	it('reads classifier records with their questions (server format 2026-11-03)', () => {
		const r = normalizeRecord({
			answer: 'pirate',
			cached: false,
			id: 2,
			kind: 'classifier',
			latency_ms: 267.25,
			n_options: 3,
			n_prompt: 113,
			n_questions: 3,
			priority: 'low',
			prob: 0.75,
			questions: [
				{ answer: 'pirate', cached: true, n_options: 3, prob: 0.75, type: 'choice' },
				{ answer: 'yes', cached: true, n_options: 2, prob: 0.69, type: 'noul' },
				{ answer: '0.8474', cached: false, n_options: 0, prob: 0.8474, type: 'score' }
			],
			status: 'done',
			t_arrival_ms: 1793675325614,
			t_end_ms: 1793675325881,
			type: 'choice'
		}) as ClassifierRecord;

		expect(r.questions).toHaveLength(3);
		expect(questionTypeKeys(r)).toEqual(['choice:3', 'noul', 'score']);
	});

	it('de-duplicates by id and arrival (ids restart with the server) and sorts newest first', () => {
		const a = completion({ id: 1, t_arrival_ms: 100 });
		const b = completion({ id: 2, t_arrival_ms: 300 });
		const restarted = completion({ id: 1, t_arrival_ms: 500 });
		const first = mergeHistory([], [a, b], 10);

		expect(first.added.map(recordKey)).toEqual([recordKey(a), recordKey(b)]);

		const second = mergeHistory(first.history, [b, a, restarted], 10);

		expect(second.added).toEqual([restarted]);
		expect(second.history.map((r) => r.t_arrival_ms)).toEqual([500, 300, 100]);
		expect(mergeHistory(second.history, [], 2).history).toHaveLength(3);
		expect(
			mergeHistory(second.history, [completion({ id: 9, t_arrival_ms: 900 })], 2).history.map(
				(r) => r.id
			)
		).toEqual([9, 1]);
	});

	it('computes the exact rates as the server does', () => {
		const r = completion();

		// decode: (n_gen - 1) / gen_ms, as predicted_per_second
		expect(decodeRate(r)).toBeCloseTo(100, 6);
		expect(prefillRate(r)).toBeCloseTo(1000, 6);
		expect(ttftMs(r)).toBe(4500);
		expect(decodeRate(completion({ n_gen: 1 }))).toBeNull();
		expect(prefillRate(completion({ n_processed: 0, prompt_ms: 0 }))).toBeNull();
		expect(ttftMs(completion({ t_first_ms: null }))).toBeNull();
	});

	it('tells a fast copy by its draft acceptance', () => {
		expect(isFastCopy(completion({ draft_accepted: 962, draft_n: 963, n_gen: 1000 }))).toBe(true);
		expect(isFastCopy(completion({ draft_accepted: 174, draft_n: 284, n_gen: 300 }))).toBe(false);
		expect(isFastCopy(completion())).toBe(false);
	});
});

describe('peaks and records', () => {
	const buckets = depthBuckets(98304);

	it('takes decode and prefill candidates after their reset, keyed by bucket bound', () => {
		const recs: RequestRecord[] = [
			completion({ n_prompt: 4025, t_end_ms: 5000 }),
			completion({ gen_ms: 500, n_prompt: 17039, t_end_ms: 9000 }),
			// a fast copy counts like any other request
			completion({
				draft_accepted: 990,
				draft_n: 1000,
				n_gen: 1000,
				n_prompt: 2300,
				t_end_ms: 9500
			}),
			// too short to count
			completion({ n_gen: 10, n_processed: 100, t_end_ms: 9600 }),
			classifier()
		];
		const c = peakCandidates(recs, buckets, 0, 6000);

		expect(c.decode.map((x) => [x.bucket, Math.round(x.value)])).toEqual([
			[8192, 100],
			[24576, 200],
			[8192, 999]
		]);
		expect(c.prefill.map((x) => x.bucket)).toEqual([24576, 8192]);
	});

	it('raises a peak only with a higher value and reports the record', () => {
		const first = updatePeaks({}, [
			{ bucket: 8192, t: 1, value: 60 },
			{ bucket: 8192, t: 2, value: 65.8 },
			{ bucket: 24576, t: 3, value: 64 }
		]);

		expect(first.peaks[8192]).toEqual({ t: 2, v: 65.8 });
		expect(first.records).toEqual([8192, 24576]);

		const second = updatePeaks(first.peaks, [
			{ bucket: 8192, t: 4, value: 61 },
			{ bucket: 24576, t: 5, value: 64.1 }
		]);

		expect(second.peaks[8192].v).toBe(65.8);
		expect(second.records).toEqual([24576]);
	});
});

describe('priority figures', () => {
	it('counts requests and averages TTFT with waiting, after the reset only', () => {
		const p = addToPriority(emptyPriority(2000), [
			completion({ priority: 'xhigh', t_end_ms: 1500 }),
			completion({ priority: 'xhigh', t_arrival_ms: 3000, t_end_ms: 9000, t_first_ms: 4400 }),
			completion({ priority: 'xhigh', t_arrival_ms: 3000, t_end_ms: 9000, t_first_ms: 3600 }),
			completion({ priority: 'low', t_arrival_ms: 3000, t_end_ms: 9000, t_first_ms: null }),
			completion({ priority: undefined, t_arrival_ms: 3000, t_end_ms: 9000, t_first_ms: 4000 }),
			classifier({ t_end_ms: 9000 })
		]);

		expect(p.byPriority.xhigh).toEqual({ n: 2, ttftN: 2, ttftSum: 2000 });
		expect(p.byPriority.low).toEqual({ n: 1, ttftN: 0, ttftSum: 0 });
		expect(p.byPriority.default).toEqual({ n: 1, ttftN: 1, ttftSum: 1000 });

		const same = emptyPriority(5);

		expect(addToPriority(same, [])).toBe(same);
	});
});

describe('question types', () => {
	it('keys by type and option count', () => {
		expect(questionTypeKey({ n_options: 2, type: 'noul' })).toBe('noul');
		expect(questionTypeKey({ n_options: 21, type: 'choice' })).toBe('choice:21');
		expect(questionTypeKey({ n_options: 0, type: 'score' })).toBe('score');
		expect(questionTypeKey({ n_options: 0, type: '' })).toBeNull();
		expect(questionTypeLabel('noul')).toBe('yes / no');
		expect(questionTypeLabel('choice:4')).toBe('choice of 4');
		expect(questionTypeLabel('score')).toBe('score');
	});

	it('orders rows yes / no, choices by option count, score', () => {
		const rows = questionTypeRows({
			'choice:3': 214,
			'choice:4': 41,
			'choice:21': 87,
			noul: 612,
			score: 118
		});

		expect(rows.map((r) => r.label)).toEqual([
			'yes / no',
			'choice of 3',
			'choice of 4',
			'choice of 21',
			'score'
		]);
		expect(rows[0].count).toBe(612);
	});

	it('adds every question of a request; error records add nothing', () => {
		const c = addToClassifier(emptyClassifier(), [
			classifier({
				questions: [
					{ n_options: 3, type: 'choice' },
					{ n_options: 2, type: 'noul' },
					{ n_options: 0, type: 'score' }
				],
				type: 'choice'
			}),
			classifier({ id: 2, n_options: 2, type: 'noul' }),
			classifier({ id: 3, status: 'error', type: '' })
		]);

		expect(c.types).toEqual({ 'choice:3': 1, noul: 2, score: 1 });
	});
});

describe('classifier speed', () => {
	it('keeps latencies, the answer count and the prompt peak (model answers only)', () => {
		const c = addToClassifier(emptyClassifier(), [
			classifier({ latency_ms: 300, n_prompt: 150, t_end_ms: 60_000 }),
			classifier({ cached: true, id: 2, latency_ms: 0.06, n_prompt: 0, t_end_ms: 90_000 }),
			classifier({ id: 3, latency_ms: 200, n_prompt: 130, t_end_ms: 120_000 })
		]);

		expect(c.n).toBe(3);
		expect(c.latencies).toEqual([300, 0.06, 200]);
		expect(c.promptPeak).toBeCloseTo(650, 6);
		expect(c.firstT).toBe(60_000);
		expect(answersPerMinute(c, 180_000)).toBeCloseTo(1.5, 6);
		expect(answersPerMinute(emptyClassifier(), 1)).toBe(0);
	});

	it('takes median and nearest-rank p95', () => {
		expect(median([3, 1, 2])).toBe(2);
		expect(median([4, 1, 2, 3])).toBe(2.5);
		expect(median([])).toBeNull();
		const v = Array.from({ length: 100 }, (_, i) => i + 1);

		expect(percentile(v, 95)).toBe(95);
		expect(percentile([5], 95)).toBe(5);
		expect(percentile([], 95)).toBeNull();
	});
});

describe('SSD cache figures', () => {
	const now = {
		bytes_read: 1977761028,
		bytes_written: 3455402452,
		enabled: true,
		evictions: 3,
		loads: 3,
		restore_ms_total: 8712.4,
		restores_timed: 3,
		saves: 5
	};

	it('subtracts the reset baseline; a counter going down means a restart', () => {
		const d = cacheDelta(now, {
			...now,
			loads: 1,
			restore_ms_total: 2000,
			restores_timed: 1,
			saves: 2
		});

		expect(d?.loads).toBe(2);
		expect(d?.saves).toBe(3);
		expect(avgRestoreMs(d)).toBeCloseTo(3356.2, 6);
		expect(cacheDelta(now, null)?.loads).toBe(3);
		expect(cacheCountersWentDown({ loads: 1 }, { loads: 3 })).toBe(true);
		expect(cacheDelta({ ...now, loads: 1, saves: 1 }, now)?.loads).toBe(1);
		expect(cacheDelta({ enabled: false }, null)).toBeNull();
		expect(avgRestoreMs(cacheDelta({ enabled: true }, null))).toBeNull();
	});

	it('adds reused tokens and prices them at the measured prefill rate', () => {
		const mem = addToCache(
			emptyCache(100),
			[
				completion({ n_cache: 37056, n_processed: 3997, prompt_ms: 5000, t_end_ms: 200 }),
				completion({ n_cache: 17035, n_processed: 61, prompt_ms: 300, t_end_ms: 300 }),
				completion({ n_cache: 999, t_end_ms: 50 })
			],
			() => 850
		);

		expect(mem.reused).toBe(37056 + 17035);
		// own rate 799.4 t/s for the first, the 850 t/s peak for the second (too few new tokens)
		expect(mem.savedMs).toBeCloseTo((37056 / (3997 / 5)) * 1000 + (17035 / 850) * 1000, 3);
	});
});
