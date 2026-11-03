/**
 * RDNA-FAST status page: constants shared by the page, the bench and the chat additions.
 * Brand colours live as CSS variables in rdna.css; the few values code needs (chart scales,
 * pill styles) are repeated here.
 */

export const RDNA_NAME = 'RDNA-FAST';
export const RDNA_TAGLINE = 'AGENT SERVER';
export const RDNA_GITHUB_URL = 'https://github.com/serrynaimo/RDNA-FAST';
/** The bench's name and version, as the speed card's footnote shows it; bump when the bench changes. */
export const RDNA_BENCH_NAME = 'rdna-bench-v1';
/** The README's section on the SSD prompt cache (--cache-disk). */
export const RDNA_SSD_CACHE_DOCS_URL = `${RDNA_GITHUB_URL}#the-ssd-prompt-cache`;

/** Link of a build's commit, from `build_info` like `b12781-707407de1`. */
export function rdnaCommitUrl(buildInfo: string | undefined | null): string | null {
	const hash = buildCommit(buildInfo);

	return hash ? `${RDNA_GITHUB_URL}/commit/${hash}` : null;
}

/** Commit hash part of `build_info` (`b12781-707407de1` -> `707407de1`), null without one. */
export function buildCommit(buildInfo: string | undefined | null): string | null {
	if (!buildInfo) return null;

	const m = /^b?\d+-([0-9a-f]{7,40})$/i.exec(buildInfo.trim());

	return m ? m[1] : null;
}

/** Build number part of `build_info` (`b12781-707407de1` -> `b12781`). */
export function buildNumber(buildInfo: string | undefined | null): string | null {
	if (!buildInfo) return null;

	const m = /^(b?\d+)/.exec(buildInfo.trim());

	return m ? (m[1].startsWith('b') ? m[1] : `b${m[1]}`) : null;
}

export const RDNA_ROUTE_STATUS = '#/status';
export const RDNA_ROUTE_ID_STATUS = '/status';

/** Poll intervals (ms), as the API contract asks. */
export const POLL_SLOTS_MS = 1000;
export const POLL_REQUESTS_MS = 1000;
export const POLL_REQUESTS_OTHER_MS = 5000;
export const POLL_PROPS_MS = 10000;
export const POLL_MODELS_MS = 5000;
/** Back-off for an endpoint the server does not have yet (404). */
export const POLL_MISSING_MS = 30000;

/** The sparking cheetah and glowing numbers only for a fast result. */
export const HOT_PREFILL_TPS = 800;
export const HOT_DECODE_TPS = 50;

/** A new record pulses its line and number this long. */
export const RECORD_PULSE_MS = 5000;

/** Requests kept per model in localStorage (the server keeps the last 50). */
export const HISTORY_CAP = 500;
/** Requests shown in the Requests tile. */
export const REQUESTS_SHOWN = 50;
/** Classifier latencies kept for the median / p95. */
export const LATENCY_SAMPLES_CAP = 500;
/** Bench runs kept per model. */
export const BENCH_RUNS_CAP = 10;

/** Smallest finished requests whose rates count for the peaks (short ones are noise). */
export const PEAK_MIN_DECODE_TOKENS = 32;
export const PEAK_MIN_PREFILL_TOKENS = 256;

/**
 * A request counts as a fast copy (n-gram drafting of text the prompt already holds) when
 * nearly every drafted token was accepted; its decode rate is not a decode peak.
 */
export const FAST_COPY_MIN_ACCEPTANCE = 0.9;
export const FAST_COPY_MIN_DRAFTS = 16;

export const PRIORITIES = ['xhigh', 'high', 'default', 'low'] as const;
export type RdnaPriority = (typeof PRIORITIES)[number];

/** Text characters used in labels. */
export const DASH = '–';
export const DOT = '·';
export const TIMES = '×';

export const STORAGE_PREFIX = 'rdna-fast:v1';
