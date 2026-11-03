/**
 * RDNA-FAST status page: number and time formatting. Every formatter takes a possibly missing value
 * and returns a dash for it, so a field the server does not send yet never breaks a tile.
 */

import { DASH } from './constants';

function ok(v: number | null | undefined): v is number {
	return typeof v === 'number' && Number.isFinite(v);
}

/** 6397 -> "6,397". */
export function fmtInt(v: number | null | undefined): string {
	return ok(v) ? Math.round(v).toLocaleString('en-US') : DASH;
}

/** Token counts in thousands: 950 -> "950", 23412 -> "23.4k", 412000 -> "412k", 3.05M -> "3.05M". */
export function fmtTokens(v: number | null | undefined): string {
	if (!ok(v)) return DASH;

	const a = Math.abs(v);

	if (a < 1000) return String(Math.round(v));

	if (a < 100_000) return `${(v / 1000).toFixed(1)}k`;

	if (a < 1_000_000) return `${Math.round(v / 1000)}k`;

	return `${(v / 1_000_000).toFixed(2)}M`;
}

/** A context size in units of 1024 tokens: 98304 -> "96k", 4096 -> "4k", 1536 -> "1.5k". */
export function fmtCtx(v: number | null | undefined): string {
	if (!ok(v) || v <= 0) return DASH;

	const k = v / 1024;

	if (k < 1) return String(Math.round(v));

	return Number.isInteger(k) ? `${k}k` : `${Math.round(k * 10) / 10}k`;
}

/** A rate in tokens per second: decode with one decimal, prefill (>= 100) as an integer. */
export function fmtRate(v: number | null | undefined, decimals?: number): string {
	if (!ok(v)) return DASH;

	const d = decimals ?? (Math.abs(v) >= 100 ? 0 : 1);

	return v.toFixed(d);
}

/** Milliseconds as seconds, minutes or hours: "13.9 s", "7.4 min", "2.1 h". */
export function fmtDuration(ms: number | null | undefined): string {
	if (!ok(ms)) return DASH;

	const s = ms / 1000;

	if (s < 60) return `${s.toFixed(1)} s`;

	if (s < 3600) return `${(s / 60).toFixed(1)} min`;

	return `${(s / 3600).toFixed(1)} h`;
}

/** A classifier answer time: "0.41 s", or "0.003 s" for the fast ones. */
export function fmtLatency(ms: number | null | undefined): string {
	if (!ok(ms)) return DASH;

	return ms < 10 ? `${(ms / 1000).toFixed(3)} s` : `${(ms / 1000).toFixed(2)} s`;
}

/** Bytes in decimal units: "137.3 GB", "1.98 GB", "512 MB". */
export function fmtBytes(v: number | null | undefined): string {
	if (!ok(v)) return DASH;

	const gb = v / 1e9;

	if (gb >= 1000) return `${(gb / 1000).toFixed(2)} TB`;

	if (gb >= 10) return `${gb.toFixed(1)} GB`;

	if (gb >= 1) return `${gb.toFixed(2)} GB`;

	const mb = v / 1e6;

	return mb >= 1 ? `${Math.round(mb)} MB` : `${Math.round(v / 1000)} KB`;
}

/** Parameter count: 27320697856 -> "27.3B", 1881825088 -> "1.9B", 300e6 -> "300M". */
export function fmtParams(v: number | null | undefined): string {
	if (!ok(v) || v <= 0) return DASH;

	if (v >= 1e9) return `${(v / 1e9).toFixed(1)}B`;

	return `${Math.round(v / 1e6)}M`;
}

function pad2(n: number): string {
	return String(n).padStart(2, '0');
}

/** Local wall-clock time "10:02:41". */
export function fmtClock(ms: number | null | undefined): string {
	if (!ok(ms)) return DASH;

	const d = new Date(ms);

	return `${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`;
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

/** "3 Nov". */
export function fmtDay(ms: number | null | undefined): string {
	if (!ok(ms)) return DASH;

	const d = new Date(ms);

	return `${d.getDate()} ${MONTHS[d.getMonth()]}`;
}

/** "09:17" today, "2 Nov 09:17" on another day. */
export function fmtSince(ms: number | null | undefined, now: number = Date.now()): string {
	if (!ok(ms)) return DASH;

	const d = new Date(ms);
	const hm = `${pad2(d.getHours())}:${pad2(d.getMinutes())}`;

	return new Date(now).toDateString() === d.toDateString() ? hm : `${fmtDay(ms)} ${hm}`;
}
