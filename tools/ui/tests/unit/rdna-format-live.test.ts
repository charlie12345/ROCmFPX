import {
	fmtBytes,
	fmtCtx,
	fmtDuration,
	fmtInt,
	fmtLatency,
	fmtParams,
	fmtRate,
	fmtTokens
} from '$lib/rdna/format';
import { LiveRates, slotPhase } from '$lib/rdna/live';
import type { RdnaSlot } from '$lib/rdna/types';
import { describe, expect, it } from 'vitest';

describe('formatting', () => {
	it('formats tokens, context and rates like the design', () => {
		expect(fmtInt(6397)).toBe('6,397');
		expect(fmtTokens(950)).toBe('950');
		expect(fmtTokens(23412)).toBe('23.4k');
		expect(fmtTokens(412000)).toBe('412k');
		expect(fmtTokens(3054473)).toBe('3.05M');
		expect(fmtCtx(98304)).toBe('96k');
		expect(fmtCtx(4096)).toBe('4k');
		expect(fmtCtx(1536)).toBe('1.5k');
		expect(fmtRate(912.4)).toBe('912');
		expect(fmtRate(61.84)).toBe('61.8');
		expect(fmtRate(61.84, 0)).toBe('62');
	});

	it('formats durations, latencies, bytes and parameters', () => {
		expect(fmtDuration(13900)).toBe('13.9 s');
		expect(fmtDuration(444000)).toBe('7.4 min');
		expect(fmtLatency(410)).toBe('0.41 s');
		expect(fmtLatency(3)).toBe('0.003 s');
		expect(fmtBytes(137281869312)).toBe('137.3 GB');
		expect(fmtBytes(1977761028)).toBe('1.98 GB');
		expect(fmtBytes(512e6)).toBe('512 MB');
		expect(fmtParams(27320697856)).toBe('27.3B');
		expect(fmtParams(1881825088)).toBe('1.9B');
		expect(fmtParams(300e6)).toBe('300M');
	});

	it('shows a dash for anything missing', () => {
		for (const f of [fmtInt, fmtTokens, fmtCtx, fmtDuration, fmtLatency, fmtBytes, fmtParams]) {
			expect(f(undefined)).toBe('–');
			expect(f(null)).toBe('–');
			expect(f(NaN)).toBe('–');
		}

		expect(fmtRate(null)).toBe('–');
	});
});

describe('live rates from /slots deltas', () => {
	const slot = (over: Partial<RdnaSlot>): RdnaSlot => ({
		id: 0,
		id_task: 7,
		is_held: false,
		is_processing: true,
		n_ctx: 98304,
		n_prompt_tokens: 0,
		n_prompt_tokens_cache: 0,
		n_prompt_tokens_processed: 0,
		next_token: [{ n_decoded: 0 }],
		priority: 'default',
		...over
	});

	it('names the phase', () => {
		expect(slotPhase(slot({ is_processing: false }))).toBe('idle');
		expect(slotPhase(slot({ is_held: true }))).toBe('held');
		expect(slotPhase(slot({}))).toBe('prefill');
		expect(slotPhase(slot({ next_token: [{ n_decoded: 3 }] }))).toBe('decode');
	});

	it('takes prefill from the first value to the last ubatch step', () => {
		const live = new LiveRates();

		expect(live.update([slot({ n_prompt_tokens_processed: 2048 })], 0)[0].prefillTps).toBeNull();
		// no step between these two polls: the rate stays unknown
		expect(live.update([slot({ n_prompt_tokens_processed: 2048 })], 1000)[0].prefillTps).toBeNull();
		expect(
			live.update([slot({ n_prompt_tokens_processed: 4096 })], 2000)[0].prefillTps
		).toBeCloseTo(1024, 6);
		// a poll without a step keeps the last rate
		expect(
			live.update([slot({ n_prompt_tokens_processed: 4096 })], 3000)[0].prefillTps
		).toBeCloseTo(1024, 6);
		expect(
			live.update([slot({ n_prompt_tokens_processed: 8192 })], 4000)[0].prefillTps
		).toBeCloseTo(1536, 6);
	});

	it('takes decode over a sliding window and restarts on a new task', () => {
		const live = new LiveRates();
		const dec = (n: number, task = 7) => slot({ id_task: task, next_token: [{ n_decoded: n }] });

		expect(live.update([dec(10)], 0)[0].decodeTps).toBeNull();
		expect(live.update([dec(70)], 1000)[0].decodeTps).toBeCloseTo(60, 6);
		expect(live.update([dec(135)], 2000)[0].decodeTps).toBeCloseTo(62.5, 6);
		expect(live.update([dec(5, 8)], 3000)[0].decodeTps).toBeNull();
		expect(live.update([slot({ is_processing: false })], 4000)[0].phase).toBe('idle');
	});
});
