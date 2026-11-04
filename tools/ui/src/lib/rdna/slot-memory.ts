/**
 * RDNA-FAST status page: what each slot showed for its latest request (owner, 2026-11-03). When the
 * request finishes, the Slots tile keeps showing it (priority, context) with its last t/s reading,
 * the value it showed right before it finished; only the status dot turns grey. The next request on
 * the slot replaces it all. Kept in memory for the page session only.
 */

import { fmtRate } from './format';
import type { SlotLive, SlotPhase } from './live';
import type { RdnaSlot } from './types';

export interface SlotMemory {
	task: number | undefined;
	priority: string;
	/** The last phase seen while the request ran. */
	phase: Exclude<SlotPhase, 'idle'>;
	ctx: number;
	processed: number;
	cached: number;
	decoded: number;
	/** The last t/s reading the slot showed, and of what. */
	lastRate: number | null;
	lastRateKind: 'decode' | 'prefill' | null;
	/** When the slot was first seen idle after the request (null while it runs). */
	endedAt: number | null;
}

/** The reading a running slot shows: decode while decoding, prefill while prefilling. */
function shownRate(l: SlotLive): { v: number; kind: 'decode' | 'prefill' } | null {
	if (l.phase === 'decode' && l.decodeTps !== null) return { kind: 'decode', v: l.decodeTps };

	if (l.phase === 'prefill' && l.prefillTps !== null) return { kind: 'prefill', v: l.prefillTps };

	return null;
}

/** Fold one /slots poll (with its live rates) into the slots' memory. */
export function updateSlotMemory(
	prev: Record<number, SlotMemory>,
	slots: readonly RdnaSlot[],
	live: readonly SlotLive[],
	t: number
): Record<number, SlotMemory> {
	const out: Record<number, SlotMemory> = { ...prev };

	slots.forEach((s, i) => {
		const l = live[i];

		if (!l) return;

		const cur = out[s.id];

		if (l.phase === 'idle') {
			// a newer request ran on the slot between two polls (or while the page was hidden, or
			// the model was loaded again): the remembered one is not the slot's latest, drop it
			if (cur && cur.task !== s.id_task) delete out[s.id];
			else if (cur && cur.endedAt === null) out[s.id] = { ...cur, endedAt: t };

			return;
		}

		const same = cur && cur.task === s.id_task && cur.endedAt === null;
		const base: SlotMemory = same
			? cur
			: {
					cached: 0,
					ctx: 0,
					decoded: 0,
					endedAt: null,
					lastRate: null,
					lastRateKind: null,
					phase: l.phase,
					priority: s.priority ?? 'default',
					processed: 0,
					task: s.id_task
				};
		const rate = shownRate(l);

		out[s.id] = {
			...base,
			cached: l.cached,
			ctx: l.ctx,
			decoded: l.decoded,
			endedAt: null,
			lastRate: rate ? rate.v : base.lastRate,
			lastRateKind: rate ? rate.kind : base.lastRateKind,
			phase: l.phase,
			priority: s.priority ?? base.priority,
			processed: l.processed
		};
	});

	return out;
}

/** The rate a finished request leaves on its slot: its last reading ("61.8 t/s"), or a dash. */
export function slotLastRateText(m: SlotMemory): string {
	if (m.lastRate === null) return '–';

	return `${fmtRate(m.lastRate, m.lastRateKind === 'prefill' ? 0 : 1)} t/s`;
}
