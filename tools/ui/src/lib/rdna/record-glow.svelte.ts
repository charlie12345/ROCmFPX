/**
 * RDNA-FAST status page: the one record-glow mechanism for every speed figure (owner, 2026-11-03).
 *
 * A figure has a key (`card:prefill`, `decode:8192`, `cls:fastest`, ...). The page seeds each key
 * with what it shows on load (stored values never glow); after that every new best value (higher, or
 * lower for a lower-is-better figure such as the fastest answer time) fires one glow of about five
 * seconds. A key the page has not seeded is seeded by its first value, silently. A tile's Reset
 * forgets its keys, so the first value after it is a new best again. The glow itself is the CSS
 * class `rf-record-bar` on a figure's bar (the bar pulses) or `rf-record` on a figure without a
 * bar (its number), in rdna.css, with a static highlight under prefers-reduced-motion.
 */

import { RECORD_PULSE_MS } from './constants';

function finite(v: number | null | undefined): number | null {
	return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

/** Is `value` better than `best` (null = nothing yet, so any value is)? */
export function isBetter(value: number, best: number | null, lowerIsBetter = false): boolean {
	if (best === null) return true;

	return lowerIsBetter ? value < best : value > best;
}

/** The bookkeeping behind the glow: the best value per key. */
export class RecordBook {
	// eslint-disable-next-line svelte/prefer-svelte-reactivity -- bookkeeping; the glow state is `until`
	private best = new Map<string, number | null>();

	clear(): void {
		this.best.clear();
	}

	/** Forget the bests of keys starting with `prefix` (a Reset): the next value is a record. */
	forget(prefix: string): void {
		for (const key of Array.from(this.best.keys())) {
			if (key.startsWith(prefix)) this.best.set(key, null);
		}
	}

	/** True when `value` is a new best of a known key; an unknown key is seeded by it. */
	observe(key: string, value: number | null | undefined, lowerIsBetter = false): boolean {
		const v = finite(value);

		if (v === null) return false;

		if (!this.best.has(key)) {
			this.best.set(key, v);

			return false;
		}

		if (!isBetter(v, this.best.get(key) ?? null, lowerIsBetter)) return false;

		this.best.set(key, v);

		return true;
	}

	/** What the page shows on load or after a reset: never a record. */
	seed(key: string, value: number | null | undefined): void {
		this.best.set(key, finite(value));
	}
}

/** The reactive glow: which keys glow until when. */
export class RecordGlow {
	until = $state.raw<Record<string, number>>({});

	private book = new RecordBook();

	clear(): void {
		this.book.clear();
		this.until = {};
	}

	forget(prefix: string): void {
		this.book.forget(prefix);

		const next: Record<string, number> = {};

		for (const [k, v] of Object.entries(this.until)) {
			if (!k.startsWith(prefix)) next[k] = v;
		}

		this.until = next;
	}

	/** Feed the shown value; a new best starts the key's glow. */
	observe(
		key: string,
		value: number | null | undefined,
		lowerIsBetter = false,
		now = Date.now()
	): boolean {
		if (!this.book.observe(key, value, lowerIsBetter)) return false;

		this.until = { ...this.until, [key]: now + RECORD_PULSE_MS };

		return true;
	}

	on(key: string, now: number): boolean {
		return (this.until[key] ?? 0) > now;
	}

	/** Drop finished glows. */
	prune(now: number): void {
		const entries = Object.entries(this.until);
		const live = entries.filter(([, v]) => v > now);

		if (live.length !== entries.length) this.until = Object.fromEntries(live);
	}

	seed(key: string, value: number | null | undefined): void {
		this.book.seed(key, value);
	}
}
