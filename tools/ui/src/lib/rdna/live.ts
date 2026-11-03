/**
 * RDNA-FAST status page: live rates from /slots polls, what the white line of the Decode and
 * Prefill rows shows (and what raises their peaks): decode over a short sliding window; prefill
 * from the first value seen in this prefill to the last ubatch step (it moves in steps of ~2048
 * tokens).
 */

import type { RdnaSlot } from './types';

export type SlotPhase = 'idle' | 'prefill' | 'decode' | 'held';

export interface SlotLive {
	phase: SlotPhase;
	/** Tokens in the slot's context now. */
	ctx: number;
	/** Prompt tokens computed so far in this request. */
	processed: number;
	cached: number;
	decoded: number;
	prefillTps: number | null;
	decodeTps: number | null;
	task: number | undefined;
}

const DECODE_WINDOW_MS = 4000;

interface Tracker {
	task: number | undefined;
	prefillFirstT: number;
	prefillFirstN: number;
	prefillLastT: number;
	prefillLastN: number;
	decodeSamples: { t: number; n: number }[];
	lastPrefillTps: number | null;
}

export function slotDecoded(s: RdnaSlot): number {
	return s.next_token?.[0]?.n_decoded ?? 0;
}

export function slotPhase(s: RdnaSlot): SlotPhase {
	if (!s.is_processing) return 'idle';

	if (s.is_held) return 'held';

	return slotDecoded(s) > 0 ? 'decode' : 'prefill';
}

/** Keeps per-slot state between polls; `update` takes one /slots answer and its arrival time. */
export class LiveRates {
	private trackers = new Map<number, Tracker>();

	reset(): void {
		this.trackers.clear();
	}

	update(slots: readonly RdnaSlot[], t: number): SlotLive[] {
		const out: SlotLive[] = [];
		const ids = new Set<number>();

		for (const s of slots) {
			ids.add(s.id);

			const phase = slotPhase(s);
			const processed = s.n_prompt_tokens_processed ?? 0;
			const decoded = slotDecoded(s);

			let tr = this.trackers.get(s.id);

			if (
				!tr ||
				tr.task !== s.id_task ||
				processed < tr.prefillLastN ||
				(decoded === 0 &&
					tr.decodeSamples.length > 0 &&
					tr.decodeSamples[tr.decodeSamples.length - 1].n > 0)
			) {
				tr = {
					decodeSamples: [],
					lastPrefillTps: null,
					prefillFirstN: processed,
					prefillFirstT: t,
					prefillLastN: processed,
					prefillLastT: t,
					task: s.id_task
				};
				this.trackers.set(s.id, tr);
			}

			let prefillTps: number | null = null;
			let decodeTps: number | null = null;

			if (phase === 'prefill' || phase === 'held') {
				if (processed > tr.prefillLastN) {
					tr.prefillLastN = processed;
					tr.prefillLastT = t;
				}

				const dt = (tr.prefillLastT - tr.prefillFirstT) / 1000;

				if (dt > 0.2 && tr.prefillLastN > tr.prefillFirstN) {
					tr.lastPrefillTps = (tr.prefillLastN - tr.prefillFirstN) / dt;
				}

				prefillTps = phase === 'prefill' ? tr.lastPrefillTps : null;
			}

			if (phase === 'decode' || (phase === 'held' && decoded > 0)) {
				tr.decodeSamples.push({ n: decoded, t });
				tr.decodeSamples = tr.decodeSamples.filter((x) => t - x.t <= DECODE_WINDOW_MS);

				const first = tr.decodeSamples[0];
				const dt = (t - first.t) / 1000;

				if (phase === 'decode' && dt >= 0.8 && decoded > first.n) {
					decodeTps = (decoded - first.n) / dt;
				}
			}

			out.push({
				cached: s.n_prompt_tokens_cache ?? 0,
				ctx: s.n_prompt_tokens ?? 0,
				decoded,
				decodeTps,
				phase,
				prefillTps,
				processed,
				task: s.id_task
			});
		}

		for (const id of Array.from(this.trackers.keys())) {
			if (!ids.has(id)) this.trackers.delete(id);
		}

		return out;
	}
}
