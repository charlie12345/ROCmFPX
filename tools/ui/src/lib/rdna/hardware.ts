/**
 * RDNA-FAST status page: hardware label and memory from /props `devices`, and the speed card's
 * label size (it shrinks for long labels so it never runs into the corner logo).
 */

import { TIMES } from './constants';
import type { RdnaDevice, RdnaProps } from './types';

function isCpu(d: RdnaDevice): boolean {
	return (d.type ?? '').toLowerCase() === 'cpu' || (d.name ?? '').toUpperCase() === 'CPU';
}

/** GPUs (type gpu / igpu; a device without a type that is not the CPU counts too). */
function isGpu(d: RdnaDevice): boolean {
	const type = (d.type ?? '').toLowerCase();

	if (type) return type === 'gpu' || type === 'igpu';

	return !isCpu(d);
}

/**
 * The devices that carry the model: its GPUs, or the CPU for a CPU-only model. Accelerator and
 * meta devices (BLAS, a split wrapper) are no hardware of their own.
 */
export function modelDevices(devices: readonly RdnaDevice[] | undefined): RdnaDevice[] {
	if (!devices || devices.length === 0) return [];

	const gpus = devices.filter(isGpu);

	return gpus.length > 0 ? gpus : devices.filter(isCpu);
}

/**
 * "AMD Radeon RX 7900 XT", "2× AMD Radeon RX 7900 XTX", "AMD Radeon RX 7900 XTX + AMD Radeon RX
 * 7800 XT"; null when the server does not report devices.
 */
export function hardwareLabel(devices: readonly RdnaDevice[] | undefined): string | null {
	const list = modelDevices(devices);

	if (list.length === 0) return null;

	const counts = new Map<string, number>();

	for (const d of list) {
		const name = (d.description || d.name || 'CPU').trim();

		counts.set(name, (counts.get(name) ?? 0) + 1);
	}

	return Array.from(counts.entries())
		.map(([name, n]) => (n > 1 ? `${n}${TIMES} ${name}` : name))
		.join(' + ');
}

export interface MemorySummary {
	used: number;
	total: number;
	/** VRAM for GPUs, RAM for the CPU. */
	kind: 'VRAM' | 'RAM';
}

/** Memory in use over the model's devices; null when the backend reports no totals. */
export function memorySummary(devices: readonly RdnaDevice[] | undefined): MemorySummary | null {
	const list = modelDevices(devices);

	let total = 0;
	let free = 0;

	for (const d of list) {
		if (!d.memory_total || d.memory_total <= 0) continue;

		total += d.memory_total;
		free += Math.max(0, Math.min(d.memory_free ?? 0, d.memory_total));
	}

	if (total <= 0) return null;

	return { kind: list.every(isCpu) ? 'RAM' : 'VRAM', total, used: total - free };
}

/** System RAM in use (bytes) and the model process's share of it. */
export interface RamSummary {
	total: number;
	used: number;
	own: number | null;
}

export function ramSummary(sm: RdnaProps['system_memory'] | undefined): RamSummary | null {
	const total = sm?.memory_total ?? 0;

	if (!sm || total <= 0) return null;

	const used = total - Math.max(0, Math.min(sm.memory_free ?? 0, total));
	const own =
		sm.process_resident && sm.process_resident > 0 ? Math.min(sm.process_resident, used) : null;

	return { own, total, used };
}

/** The design's label sizes (in cqw of the card) by label length. */
export function designLabelSize(label: string): number {
	const n = label.length;

	return n <= 16 ? 3.8 : n <= 22 ? 3.1 : 2.6;
}

/**
 * Font size (cqw) of the card's hardware label: the design size, smaller when the label at that
 * size would be wider than its column (39cqw, 4% kept free). `widthAt100px` is the label's width
 * rendered at font-size 100px.
 */
export function cardLabelSize(label: string, widthAt100px: number | null): number {
	const design = designLabelSize(label);

	if (!widthAt100px || widthAt100px <= 0) return design;

	// width at f cqw = f * widthAt100px / 100 (cqw); the column holds 39 * 0.96 cqw
	const fit = (39 * 0.96 * 100) / widthAt100px;

	return Math.max(1.2, Math.min(design, Math.floor(fit * 100) / 100));
}
