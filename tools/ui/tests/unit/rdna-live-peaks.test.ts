import { SIDEBAR_ACTIONS_ITEMS } from '$lib/constants/ui.constants';
import { BENCH_PRIORITY, heldText, othersOnModel } from '$lib/rdna/bench/runner.svelte';
import { CARD_EMPTY_LINE, cardFigures, cardFooter, cardSource } from '$lib/rdna/card';
import { depthBuckets } from '$lib/rdna/depth';
import { peakCandidates } from '$lib/rdna/history';
import type { SlotLive } from '$lib/rdna/live';
import { isBetter, RecordBook, RecordGlow } from '$lib/rdna/record-glow.svelte';
import { updateSlotMemory } from '$lib/rdna/slot-memory';
import { StatusStore } from '$lib/rdna/status-store.svelte';
import { loadCopyBoost } from '$lib/rdna/storage';
import type { CompletionRecord, RdnaSlot } from '$lib/rdna/types';
import { describe, expect, it } from 'vitest';

function completion(over: Partial<CompletionRecord>): CompletionRecord {
	return {
		draft_accepted: 0,
		draft_n: 0,
		gen_ms: 1000,
		id: 1,
		kind: 'completion',
		n_cache: 0,
		n_gen: 101,
		n_processed: 4000,
		n_prompt: 4000,
		prompt_ms: 4000,
		status: 'done',
		t_arrival_ms: 1000,
		t_end_ms: 7000,
		t_first_ms: 5000,
		...over
	};
}

describe('speed card: finished bench runs only', () => {
	it('takes the bench figures and knows when there are none', () => {
		expect(cardFigures({ decode_4k: 65.8, edit: 311, prefill_4k: 1002 })).toEqual({
			copy: 311,
			decode: 65.8,
			empty: false,
			prefill: 1002
		});
		expect(cardFigures({ session: 96.7 })).toEqual({
			copy: undefined,
			decode: undefined,
			empty: true,
			prefill: undefined
		});
		expect(CARD_EMPTY_LINE).toBe('Run the bench to fill in this card');
	});

	it('names the bench run in the footer', () => {
		expect(cardSource(2)).toBe('rdna-bench-v1, best of the last two runs');
		expect(cardSource(1)).toBe('rdna-bench-v1, one run');
		expect(cardSource(0)).toBe('no rdna-bench-v1 run yet');
		expect(
			cardFooter('Swift-1.5-Qwen3.8-27B IQ4_XS', 98304, 'b12781-707407de1', cardSource(2))
		).toBe(
			'Swift-1.5-Qwen3.8-27B IQ4_XS · 96k context · b12781 · rdna-bench-v1, best of the last two runs'
		);
	});

	it('counts fast copies in the decode peaks (the owner accepts text copies setting them)', () => {
		const buckets = depthBuckets(98304);
		const recs = [
			completion({ gen_ms: 1500, id: 1 }),
			completion({ draft_accepted: 990, draft_n: 1000, gen_ms: 3300, id: 2, n_gen: 1000 })
		];
		const c = peakCandidates(recs, buckets, 0, 0);

		expect(c.decode.map((x) => Math.round(x.value))).toEqual([67, 303]);
	});
});

describe('record glow (one mechanism for every speed figure)', () => {
	it('fires once per new best, higher is better by default', () => {
		const book = new RecordBook();

		book.seed('prefill:8192', 1002);
		expect(book.observe('prefill:8192', 1002)).toBe(false);
		expect(book.observe('prefill:8192', 990)).toBe(false);
		expect(book.observe('prefill:8192', 1010)).toBe(true);
		// the same value again is no new best
		expect(book.observe('prefill:8192', 1010)).toBe(false);
		expect(book.observe('prefill:8192', 1011)).toBe(true);
	});

	it('treats lower-is-better figures correctly', () => {
		const book = new RecordBook();

		book.seed('cls:fastest', 300);
		expect(book.observe('cls:fastest', 310, true)).toBe(false);
		expect(book.observe('cls:fastest', 280, true)).toBe(true);
		expect(book.observe('cls:fastest', 280, true)).toBe(false);
		expect(isBetter(5, 6, true)).toBe(true);
		expect(isBetter(5, null, true)).toBe(true);
		expect(isBetter(5, 6)).toBe(false);
	});

	it('does not glow for stored values on load; an empty figure glows on its first value', () => {
		const book = new RecordBook();

		// stored peak seeded on load: no glow, and an unseeded key seeds itself silently
		book.seed('decode:8192', 65.8);
		book.seed('decode:24576', null);
		expect(book.observe('decode:8192', 65.8)).toBe(false);
		expect(book.observe('decode:98304', 50)).toBe(false);
		// a figure that had nothing yet: its first value is a new best
		expect(book.observe('decode:24576', 61)).toBe(true);
		expect(book.observe('decode:24576', null)).toBe(false);
	});

	it('forgets a tile’s bests on Reset and drops finished glows', () => {
		const glow = new RecordGlow();

		glow.seed('decode:8192', 70);
		glow.seed('prefill:8192', 990);
		glow.forget('decode:');
		expect(glow.observe('decode:8192', 50, false, 1000)).toBe(true);
		expect(glow.observe('prefill:8192', 900, false, 1000)).toBe(false);
		expect(glow.on('decode:8192', 2000)).toBe(true);
		expect(glow.on('decode:8192', 7000)).toBe(false);
		glow.prune(7000);
		expect(glow.until).toEqual({});
	});
});

describe('status store: the simple peak rule', () => {
	function liveEntry(task: number, ctx: number, decodeTps: number | null): SlotLive {
		return {
			cached: 0,
			ctx,
			decoded: 100,
			decodeTps,
			phase: 'decode',
			prefillTps: null,
			processed: 0,
			task
		};
	}

	/** A store on model `id`; `boost` = the Decode tile's "Text copy boost" (null: leave as loaded). */
	function makeStore(id = 'm', boost: boolean | null = true): StatusStore {
		const store = new StatusStore();

		store.entries = [{ id, meta: { n_ctx: 98304 }, status: { value: 'loaded' } }];
		// selecting loads (here: empty) memory and seeds the glow with what the page shows
		store.select(id);

		if (boost !== null) store.setCopyBoost(boost);

		return store;
	}

	type Internals = { raisePeaksFromLive(live: SlotLive[], t: number): void };

	const poll = (store: StatusStore, v: number | null, t = Date.now(), ctx = 5000, task = 7) =>
		(store as unknown as Internals).raisePeaksFromLive([liveEntry(task, ctx, v)], t);

	it('takes the white line as the peak when it goes past it, exactly, with the glow', () => {
		const store = makeStore();

		poll(store, 70);
		expect(store.decodeRows[0].peak).toBe(70);
		expect(store.decodeRows[0].record).toBe(true);

		const first = store.glow.until['decode:8192'];

		// below the bar: the bar stays, no new glow
		poll(store, 69, Date.now() + 1);
		expect(store.decodeRows[0].peak).toBe(70);
		expect(store.glow.until['decode:8192']).toBe(first);

		// past it again: the bar jumps to exactly the line
		poll(store, 91.37, Date.now() + 2);
		expect(store.decodeRows[0].peak).toBe(91.37);
		expect(store.glow.until['decode:8192']).toBeGreaterThanOrEqual(first);
		// the bucket is the request's current depth
		poll(store, 50, Date.now() + 3, 30_000);
		expect(store.decodeRows[2].peak).toBe(50);
		// the speed card stays with the bench (none here)
		expect(store.card.empty).toBe(true);
	});

	it('raises the peak by a finished request’s exact rate the polls never caught, never lowers it', () => {
		const store = makeStore();

		poll(store, 80);
		// a request between two polls ran faster: its exact rate raises the bar
		store.ingest('m', [
			completion({ gen_ms: 1000, id: 7, n_prompt: 5000, t_end_ms: Date.now() + 10 })
		]);
		store.ingest('m', [
			completion({ gen_ms: 800, id: 8, n_prompt: 5000, t_end_ms: Date.now() + 20 })
		]);
		expect(store.decodeRows[0].peak).toBeCloseTo(125, 6);
		// a slower one changes nothing
		store.ingest('m', [
			completion({ gen_ms: 2000, id: 9, n_prompt: 5000, t_end_ms: Date.now() + 30 })
		]);
		expect(store.decodeRows[0].peak).toBeCloseTo(125, 6);
		// Reset clears it
		store.resetDecode();
		expect(store.decodeRows[0].peak).toBeNull();
	});
});

describe('status store: Text copy boost off (decode peaks without fast copies)', () => {
	function liveEntry(task: number, ctx: number, decodeTps: number): SlotLive {
		return {
			cached: 0,
			ctx,
			decoded: 100,
			decodeTps,
			phase: 'decode',
			prefillTps: null,
			processed: 0,
			task
		};
	}

	type Internals = { raisePeaksFromLive(live: SlotLive[], t: number): void };

	let n = 0;

	function makeStore(): StatusStore {
		const store = new StatusStore();
		const id = `nc-${++n}`;

		store.entries = [{ id, meta: { n_ctx: 98304 }, status: { value: 'loaded' } }];
		store.select(id);
		// the page's first answer brings history: no glow for it
		store.ingest(id, [completion({ gen_ms: 4000, id: 1, n_prompt: 5000, t_end_ms: 1000 })]);

		return store;
	}

	const poll = (store: StatusStore, task: number, v: number, t: number, ctx = 5000) =>
		(store as unknown as Internals).raisePeaksFromLive([liveEntry(task, ctx, v)], t);

	it('is the default: the bar stays while the request runs', () => {
		const store = makeStore();
		const t = Date.now();

		expect(store.copyBoost).toBe(false);

		poll(store, 7, 91.37, t + 1);
		poll(store, 7, 80, t + 2);
		// the white line moved past the bar; the bar did not follow
		expect(store.decodeRows[0].peak).toBe(25);
		expect(store.decodeRows[0].record).toBe(false);
	});

	it('commits the request’s best reading when it finishes as no fast copy, with the glow', () => {
		const store = makeStore();
		const t = Date.now();

		poll(store, 7, 91.37, t + 1);
		poll(store, 7, 80, t + 2);
		// its record: 50 t/s overall, no fast copy; the best reading (91.37) is the record
		store.ingest(store.selectedId as string, [
			completion({ gen_ms: 2000, id: 7, n_prompt: 5000, t_end_ms: t + 3 })
		]);
		expect(store.decodeRows[0].peak).toBe(91.37);
		expect(store.decodeRows[0].record).toBe(true);
		expect(store.glow.until[`decode-nc:8192`]).toBeGreaterThan(0);
	});

	it('takes the exact rate when it is higher than every reading', () => {
		const store = makeStore();
		const t = Date.now();

		poll(store, 9, 60, t + 1);
		store.ingest(store.selectedId as string, [
			completion({ gen_ms: 800, id: 9, n_prompt: 5000, t_end_ms: t + 2 })
		]);
		expect(store.decodeRows[0].peak).toBeCloseTo(125, 6);
	});

	it('never takes a fast copy, and the toggle switches between the two sets at once', () => {
		const store = makeStore();
		const t = Date.now();

		poll(store, 8, 303, t + 1);
		store.ingest(store.selectedId as string, [
			completion({
				draft_accepted: 990,
				draft_n: 1000,
				gen_ms: 3300,
				id: 8,
				n_gen: 1000,
				n_prompt: 5000,
				t_end_ms: t + 2
			})
		]);
		// without copies: unchanged (the history's 25 t/s)
		expect(store.decodeRows[0].peak).toBe(25);

		// with copies: the bar followed the line to 303
		const glows = { ...store.glow.until };

		store.setCopyBoost(true);
		expect(store.decodeRows[0].peak).toBe(303);
		// switching shows the other set and starts no glow of its own
		expect(store.glow.until).toEqual(glows);
		store.setCopyBoost(false);
		expect(store.decodeRows[0].peak).toBe(25);
	});

	it('never lowers a committed record; Reset clears both sets', () => {
		const store = makeStore();
		const id = store.selectedId as string;
		const t = Date.now();

		poll(store, 7, 91.37, t + 1);
		store.ingest(id, [completion({ gen_ms: 2000, id: 7, n_prompt: 5000, t_end_ms: t + 2 })]);
		poll(store, 10, 70, t + 3);
		store.ingest(id, [completion({ gen_ms: 2000, id: 10, n_prompt: 5000, t_end_ms: t + 4 })]);
		expect(store.decodeRows[0].peak).toBe(91.37);

		store.resetDecode();
		expect(store.decodeRows[0].peak).toBeNull();
		store.setCopyBoost(true);
		expect(store.decodeRows[0].peak).toBeNull();
	});

	it('keeps the toggle per model', () => {
		const store = makeStore();
		const id = store.selectedId as string;

		store.setCopyBoost(true);

		const again = new StatusStore();

		again.entries = [{ id, meta: { n_ctx: 98304 }, status: { value: 'loaded' } }];
		again.select(id);
		expect(again.copyBoost).toBe(loadCopyBoost(id));
		expect(loadCopyBoost('never-set')).toBe(false);
	});
});

describe('status store: the shared glow on every speed figure', () => {
	function makeStore(): StatusStore {
		const store = new StatusStore();

		store.entries = [{ id: 'm', meta: { n_ctx: 98304 }, status: { value: 'loaded' } }];
		store.select('m');

		return store;
	}

	it('does not glow for the history of the first answer, then glows each new best of a row', () => {
		const store = makeStore();

		// the page's first /requests answer: history, shown without a glow
		store.ingest('m', [completion({ id: 1, n_prompt: 5000, t_end_ms: 1000 })]);
		expect(store.decodeRows[0].peak).toBeCloseTo(100, 6);
		expect(store.glow.on('decode:8192', Date.now())).toBe(false);
		expect(store.decodeRows[0].record).toBe(false);

		// a faster request: the row glows
		store.ingest('m', [completion({ gen_ms: 800, id: 2, n_prompt: 5000, t_end_ms: 2000 })]);
		expect(store.decodeRows[0].peak).toBeCloseTo(125, 6);
		expect(store.glow.on('decode:8192', Date.now())).toBe(true);
		expect(store.decodeRows[0].record).toBe(true);
		// the prefill rate did not improve: no glow there; the card never follows live requests
		expect(store.glow.on('prefill:8192', Date.now())).toBe(false);
		expect(store.card.empty).toBe(true);
	});

	it('glows the classifier’s fastest answer when it gets lower', () => {
		const store = makeStore();
		const answer = (id: number, latency: number, cached = false) => ({
			cached,
			id,
			kind: 'classifier' as const,
			latency_ms: latency,
			n_options: 2,
			n_prompt: cached ? 0 : 110,
			status: 'done',
			t_arrival_ms: id * 1000,
			t_end_ms: id * 1000 + latency,
			type: 'noul'
		});

		store.ingest('m', [answer(1, 300)]);
		expect(store.classifier.fastest).toBe(300);
		// a cached answer is no model speed
		store.ingest('m', [answer(2, 0.06, true)]);
		expect(store.classifier.fastest).toBe(300);
		expect(store.glow.on('cls:fastest', Date.now())).toBe(false);
		store.ingest('m', [answer(3, 250)]);
		expect(store.classifier.fastest).toBe(250);
		expect(store.glow.on('cls:fastest', Date.now())).toBe(true);
	});
});

describe('slots keep their finished request', () => {
	const slot = (task: number, busy: boolean, priority = 'xhigh'): RdnaSlot => ({
		id: 1,
		id_task: task,
		is_processing: busy,
		priority
	});
	const live = (over: Partial<SlotLive>): SlotLive => ({
		cached: 17044,
		ctx: 23441,
		decoded: 0,
		decodeTps: null,
		phase: 'prefill',
		prefillTps: 900,
		processed: 6397,
		task: 9,
		...over
	});

	it('keeps the pill and the context after the request', () => {
		let m = updateSlotMemory({}, [slot(9, true)], [live({})], 1000);

		m = updateSlotMemory(
			m,
			[slot(9, true)],
			[live({ decoded: 120, decodeTps: 88.1, phase: 'decode' })],
			2000
		);
		m = updateSlotMemory(
			m,
			[slot(9, true)],
			[live({ decoded: 210, decodeTps: 92.4, phase: 'decode' })],
			3000
		);
		m = updateSlotMemory(
			m,
			[slot(9, true)],
			[live({ decoded: 290, decodeTps: 80.2, phase: 'decode' })],
			4000
		);
		expect(m[1].endedAt).toBeNull();

		// finished: the slot reports idle with the default priority, the memory keeps the request
		m = updateSlotMemory(m, [slot(9, false, 'default')], [live({ phase: 'idle' })], 5000);
		expect(m[1]).toMatchObject({
			ctx: 23441,
			decoded: 290,
			endedAt: 5000,
			phase: 'decode',
			priority: 'xhigh'
		});

		// still idle: nothing changes
		expect(updateSlotMemory(m, [slot(9, false)], [live({ phase: 'idle' })], 6000)[1].endedAt).toBe(
			5000
		);

		// the next request replaces it
		const next = updateSlotMemory(
			m,
			[slot(10, true, 'low')],
			[live({ prefillTps: 950, task: 10 })],
			7000
		);

		expect(next[1]).toMatchObject({ endedAt: null, priority: 'low', task: 10 });
	});

	it('forgets the request when a newer one ran on the slot unseen', () => {
		let m = updateSlotMemory(
			{},
			[slot(9, true)],
			[live({ decoded: 120, decodeTps: 88.1, phase: 'decode' })],
			1000
		);

		m = updateSlotMemory(m, [slot(9, false)], [live({ phase: 'idle' })], 2000);
		expect(m[1]).toMatchObject({ endedAt: 2000, task: 9 });

		// task 12 started and finished between two polls: the slot reports idle with it
		m = updateSlotMemory(m, [slot(12, false)], [live({ phase: 'idle', task: 12 })], 3000);
		expect(m[1]).toBeUndefined();
	});
});

describe('bench at priority xhigh', () => {
	it('runs at xhigh and counts other requests on the model as held', () => {
		expect(BENCH_PRIORITY).toBe('xhigh');

		const s = (busy: boolean, held = false): RdnaSlot => ({
			id: 0,
			is_held: held,
			is_processing: busy
		});

		expect(othersOnModel([s(true), s(false)])).toBe(0);
		expect(othersOnModel([s(true), s(true, true)])).toBe(1);
		expect(othersOnModel([])).toBe(0);
		expect(heldText(0)).toBe('');
		expect(heldText(1)).toBe(' · 1 other request held');
		expect(heldText(3)).toBe(' · 3 other requests held');
	});
});

describe('sidebar', () => {
	it('lists the Status page before Settings', () => {
		const tips = SIDEBAR_ACTIONS_ITEMS.map((i) => i.tooltip);
		const status = SIDEBAR_ACTIONS_ITEMS.find((i) => i.tooltip === 'Status');

		expect(tips.indexOf('Status')).toBe(tips.indexOf('Settings') - 1);
		expect(status?.route).toBe('#/status');
		expect(status?.activeRouteId).toBe('/status');
	});
});
