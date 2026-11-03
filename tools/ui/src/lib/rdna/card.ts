/**
 * RDNA-FAST speed card figures: finished bench runs only (owner, 2026-11-03), the best of the last
 * two: prefill = the cold 4k prompt, decode = decode at 4k, copy = the best code edit. Live requests
 * never change the card; a running bench changes it only once it has finished. Without a finished
 * run the card shows no numbers, just a line asking for a bench run.
 */

import type { BenchValues } from './bench/types';
import { buildNumber, HOT_DECODE_TPS, HOT_PREFILL_TPS, RDNA_BENCH_NAME } from './constants';
import { fmtCtx } from './format';

/**
 * The card's cheetah: snoozing before the model has any bench value, sparking for a fast result
 * (>= 800 t/s prefill and >= 50 t/s decode), walking otherwise.
 */
export type Cheetah = 'sleep' | 'walk' | 'spark';

/** The pictures under static/rdna/ (about 950 x 705, the cheetah on the left half). */
export const CHEETAH_FILES: Record<Cheetah, string> = {
	sleep: 'cheetah-sleep.jpg',
	spark: 'cheetah-plate.jpg',
	walk: 'cheetah-walk.jpg'
};

export function cheetahState(f: {
	prefill: number | undefined;
	decode: number | undefined;
	copy: number | undefined;
}): Cheetah {
	const has = (v: number | undefined) => typeof v === 'number' && Number.isFinite(v);

	if (!has(f.prefill) && !has(f.decode) && !has(f.copy)) return 'sleep';

	return has(f.prefill) &&
		has(f.decode) &&
		(f.prefill as number) >= HOT_PREFILL_TPS &&
		(f.decode as number) >= HOT_DECODE_TPS
		? 'spark'
		: 'walk';
}

/** What the card says where its numbers would be, before the first finished bench run. */
export const CARD_EMPTY_LINE = 'Run the bench to fill in this card';

export interface CardFigures {
	prefill: number | undefined;
	decode: number | undefined;
	copy: number | undefined;
	/** No finished bench run has any of the figures: the card's empty state. */
	empty: boolean;
}

function finite(v: number | undefined): number | undefined {
	return typeof v === 'number' && Number.isFinite(v) ? v : undefined;
}

export function cardFigures(bench: BenchValues): CardFigures {
	const prefill = finite(bench.prefill_4k);
	const decode = finite(bench.decode_4k);
	const copy = finite(bench.edit);

	return {
		copy,
		decode,
		empty: prefill === undefined && decode === undefined && copy === undefined,
		prefill
	};
}

/** The bench run(s) the card shows, for its footer. */
export function cardSource(benchRuns: number): string {
	if (benchRuns >= 2) return `${RDNA_BENCH_NAME}, best of the last two runs`;

	if (benchRuns === 1) return `${RDNA_BENCH_NAME}, one run`;

	return `no ${RDNA_BENCH_NAME} run yet`;
}

/** The card's footer: model, context, build of the newest run, source. */
export function cardFooter(
	modelLabel: string,
	nCtx: number | null,
	buildInfo: string | null,
	source: string
): string {
	const parts = [modelLabel];

	if (nCtx) parts.push(`${fmtCtx(nCtx)} context`);

	const build = buildNumber(buildInfo);

	if (build) parts.push(build);

	parts.push(source);

	return parts.join(' · ');
}
