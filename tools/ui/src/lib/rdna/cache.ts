/**
 * RDNA-FAST status page: the SSD cache tile's figures. The server's counters run since its start;
 * the page subtracts the values it saw at the last reset (a counter below its baseline means the
 * server restarted, and then everything counts from zero again). Reused tokens and the prefill
 * time they saved add up from finished requests.
 */

import { PEAK_MIN_PREFILL_TOKENS } from './constants';
import { prefillRate, recordDepth } from './history';
import type { CacheMemory, RdnaPromptCacheDisk, RequestRecord } from './types';

const COUNTERS = [
	'saves',
	'loads',
	'evictions',
	'bytes_read',
	'bytes_written',
	'restore_ms_total',
	'restores_timed'
] as const;

export type CacheCounter = (typeof COUNTERS)[number];

/** A counter below its baseline: the server restarted since the baseline was taken. */
export function cacheCountersWentDown(
	now: RdnaPromptCacheDisk,
	baseline: RdnaPromptCacheDisk
): boolean {
	return COUNTERS.some(
		(k) =>
			typeof now[k] === 'number' &&
			typeof baseline[k] === 'number' &&
			(now[k] as number) < (baseline[k] as number)
	);
}

/** Counters since the baseline; null when the server does not report them. */
export function cacheDelta(
	now: RdnaPromptCacheDisk | undefined | null,
	baseline: RdnaPromptCacheDisk | null
): Record<CacheCounter, number> | null {
	if (!now || now.enabled === false) return null;

	const restarted = !!baseline && cacheCountersWentDown(now, baseline);
	const out = {} as Record<CacheCounter, number>;

	for (const k of COUNTERS) {
		const cur = typeof now[k] === 'number' ? (now[k] as number) : 0;
		const b =
			!baseline || restarted ? 0 : typeof baseline[k] === 'number' ? (baseline[k] as number) : 0;

		out[k] = Math.max(0, cur - b);
	}

	return out;
}

/** Average restore time (ms) from a delta, null before the first timed restore. */
export function avgRestoreMs(d: Record<CacheCounter, number> | null): number | null {
	if (!d || d.restores_timed <= 0) return null;

	return d.restore_ms_total / d.restores_timed;
}

/**
 * Add finished requests' cache reuse. A request's own prefill rate prices its reused tokens when it
 * computed enough new ones, else `rateAt(depth)` (the prefill peak at that depth).
 */
export function addToCache(
	mem: CacheMemory,
	records: readonly RequestRecord[],
	rateAt: (depth: number) => number | null
): CacheMemory {
	let reused = mem.reused;
	let savedMs = mem.savedMs;
	let changed = false;

	for (const r of records) {
		if (r.kind !== 'completion' || r.t_end_ms <= mem.at || !r.n_cache || r.n_cache <= 0) continue;

		reused += r.n_cache;
		changed = true;

		const own = (r.n_processed ?? 0) >= PEAK_MIN_PREFILL_TOKENS ? prefillRate(r) : null;
		const rate = own ?? rateAt(recordDepth(r));

		if (rate && rate > 0) savedMs += (r.n_cache / rate) * 1000;
	}

	return changed ? { ...mem, reused, savedMs } : mem;
}
