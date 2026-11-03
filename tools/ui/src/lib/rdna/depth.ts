/**
 * RDNA-FAST status page: context depth buckets of the Decode and Prefill tiles. Three fixed rows
 * (owner, 2026-11-03): "up to 8k", "8k to 24k" and "24k+" (everything from 24k on). A model whose
 * context ends before a bucket starts does not show that bucket. Peaks are stored under a bucket's
 * upper bound (`hi`); older stores used more buckets and are folded in by migratePeakMap.
 */

import type { PeakMap } from './types';

export interface DepthBucket {
	/** First token depth of the bucket. */
	lo: number;
	/** Depth the bucket ends at (its key in the stored peaks). */
	hi: number;
	label: string;
}

/** The open last bucket's upper bound: further than any context. */
export const DEPTH_OPEN_HI = 1_000_000_000;

const BUCKETS: readonly DepthBucket[] = [
	{ hi: 8 * 1024, label: 'up to 8k', lo: 0 },
	{ hi: 24 * 1024, label: '8k to 24k', lo: 8 * 1024 },
	{ hi: DEPTH_OPEN_HI, label: '24k+', lo: 24 * 1024 }
];

/** The buckets a context of `nCtx` tokens can reach (all three when unknown). */
export function depthBuckets(nCtx: number | null | undefined): DepthBucket[] {
	if (!nCtx || nCtx <= 0) return BUCKETS.slice();

	return BUCKETS.filter((b) => b.lo < nCtx);
}

/** Index of the bucket a depth falls in; depths beyond the buckets go to the last one. */
export function bucketIndex(depth: number, buckets: readonly DepthBucket[]): number {
	if (buckets.length === 0) return -1;

	for (let i = 0; i < buckets.length; i++) {
		if (depth < buckets[i].hi) return i;
	}

	return buckets.length - 1;
}

/**
 * Fold peaks stored under any older bucket bound (24k-48k, 48k-96k, ...) into today's buckets,
 * keeping the highest value per bucket.
 */
export function migratePeakMap(map: PeakMap): PeakMap {
	const out: PeakMap = {};

	for (const [key, entry] of Object.entries(map)) {
		const bound = Number(key);

		if (!Number.isFinite(bound) || !entry || typeof entry.v !== 'number') continue;

		const hi = BUCKETS[bucketIndex(bound - 1, BUCKETS)].hi;
		const cur = out[hi];

		if (!cur || entry.v > cur.v) out[hi] = entry;
	}

	return out;
}

/** A round upper end for a bar scale, a bit above the largest value. */
export function niceScale(max: number): number {
	if (!Number.isFinite(max) || max <= 0) return 1;

	const target = max * 1.1;
	const pow = Math.pow(10, Math.floor(Math.log10(target)));

	for (const step of [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) {
		if (step * pow >= target) return step * pow;
	}

	return 10 * pow;
}
