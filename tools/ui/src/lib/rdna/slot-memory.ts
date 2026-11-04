/**
 * RDNA-FAST status page: what each slot showed for its latest request (owner, 2026-11-03). When the
 * request finishes, the Slots tile keeps showing it (priority, context); only the status dot turns
 * grey, and the t/s goes (owner, 2026-11-04: an idle slot shows no rate). The next request on the
 * slot replaces it all. Kept in memory for the page session only.
 */

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
	/** When the slot was first seen idle after the request (null while it runs). */
	endedAt: number | null;
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
					phase: l.phase,
					priority: s.priority ?? 'default',
					processed: 0,
					task: s.id_task
				};

		out[s.id] = {
			...base,
			cached: l.cached,
			ctx: l.ctx,
			decoded: l.decoded,
			endedAt: null,
			phase: l.phase,
			priority: s.priority ?? base.priority,
			processed: l.processed
		};
	});

	return out;
}
