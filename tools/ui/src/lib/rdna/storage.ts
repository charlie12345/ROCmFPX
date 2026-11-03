/**
 * RDNA-FAST status page: everything the page remembers lives in localStorage, one key per model
 * and part, so the bench (which writes `bench`) and the page (everything else) never overwrite
 * each other's data. Every access is guarded: a private window or blocked storage just forgets.
 */

import { HISTORY_CAP, STORAGE_PREFIX } from './constants';
import { migratePeakMap } from './depth';
import type {
	CacheMemory,
	ClassifierMemory,
	PeaksMemory,
	PriorityMemory,
	RequestRecord
} from './types';

export type ModelPart =
	| 'history'
	| 'peaks'
	| 'prio'
	| 'cache'
	| 'classifier'
	| 'hw'
	| 'bench'
	| 'copyBoost';

export function storageKey(modelId: string, part: ModelPart): string {
	return `${STORAGE_PREFIX}:model:${modelId}:${part}`;
}

export function readJson<T>(key: string): T | null {
	try {
		if (typeof localStorage === 'undefined') return null;

		const raw = localStorage.getItem(key);

		return raw ? (JSON.parse(raw) as T) : null;
	} catch {
		return null;
	}
}

export function writeJson(key: string, value: unknown): void {
	try {
		if (typeof localStorage === 'undefined') return;

		if (value === null || value === undefined) {
			localStorage.removeItem(key);
		} else {
			localStorage.setItem(key, JSON.stringify(value));
		}
	} catch {
		// quota or blocked storage: keep going with what is in memory
	}
}

export function readPart<T>(modelId: string, part: ModelPart): T | null {
	return readJson<T>(storageKey(modelId, part));
}

export function writePart(modelId: string, part: ModelPart, value: unknown): void {
	writeJson(storageKey(modelId, part), value);
}

function isObject(v: unknown): v is Record<string, unknown> {
	return !!v && typeof v === 'object' && !Array.isArray(v);
}

export function loadHistory(modelId: string): RequestRecord[] {
	const v = readPart<RequestRecord[]>(modelId, 'history');

	return Array.isArray(v)
		? v.filter((r) => isObject(r) && typeof r.id === 'number').slice(0, HISTORY_CAP)
		: [];
}

export function emptyPeaks(): PeaksMemory {
	return { decode: {}, decodeNoCopy: {}, decodeResetAt: 0, prefill: {}, prefillResetAt: 0 };
}

export function loadPeaks(modelId: string): PeaksMemory {
	const v = readPart<PeaksMemory>(modelId, 'peaks');

	if (!isObject(v)) return emptyPeaks();

	return {
		decode: isObject(v.decode) ? migratePeakMap(v.decode as PeaksMemory['decode']) : {},
		decodeNoCopy: isObject(v.decodeNoCopy)
			? migratePeakMap(v.decodeNoCopy as PeaksMemory['decodeNoCopy'])
			: {},
		decodeResetAt: typeof v.decodeResetAt === 'number' ? v.decodeResetAt : 0,
		prefill: isObject(v.prefill) ? migratePeakMap(v.prefill as PeaksMemory['prefill']) : {},
		prefillResetAt: typeof v.prefillResetAt === 'number' ? v.prefillResetAt : 0
	};
}

export function emptyPriority(resetAt = 0): PriorityMemory {
	return { byPriority: {}, resetAt };
}

export function loadPriority(modelId: string): PriorityMemory {
	const v = readPart<PriorityMemory>(modelId, 'prio');

	if (!isObject(v) || !isObject(v.byPriority)) return emptyPriority();

	return { byPriority: v.byPriority, resetAt: typeof v.resetAt === 'number' ? v.resetAt : 0 };
}

export function emptyCache(at = 0): CacheMemory {
	return { at, counters: null, reused: 0, savedMs: 0 };
}

export function loadCache(modelId: string): CacheMemory {
	const v = readPart<CacheMemory>(modelId, 'cache');

	if (!isObject(v) || typeof v.at !== 'number') return emptyCache();

	return {
		at: v.at,
		counters: isObject(v.counters) ? v.counters : null,
		reused: typeof v.reused === 'number' ? v.reused : 0,
		savedMs: typeof v.savedMs === 'number' ? v.savedMs : 0
	};
}

export function emptyClassifier(): ClassifierMemory {
	return {
		fastest: null,
		firstT: null,
		latencies: [],
		n: 0,
		promptPeak: null,
		speedResetAt: 0,
		types: {},
		typesResetAt: 0
	};
}

export function loadClassifier(modelId: string): ClassifierMemory {
	const v = readPart<ClassifierMemory>(modelId, 'classifier');

	if (!isObject(v)) return emptyClassifier();

	const d = emptyClassifier();

	return {
		fastest: typeof v.fastest === 'number' ? v.fastest : null,
		firstT: typeof v.firstT === 'number' ? v.firstT : d.firstT,
		latencies: Array.isArray(v.latencies) ? v.latencies.filter((x) => typeof x === 'number') : [],
		n: typeof v.n === 'number' ? v.n : 0,
		promptPeak: typeof v.promptPeak === 'number' ? v.promptPeak : null,
		speedResetAt: typeof v.speedResetAt === 'number' ? v.speedResetAt : 0,
		types: isObject(v.types) ? (v.types as Record<string, number>) : {},
		typesResetAt: typeof v.typesResetAt === 'number' ? v.typesResetAt : 0
	};
}

/**
 * The GPU name /props last reported for a model, kept for the browser tab (sessionStorage, owner
 * 2026-11-03): a reload shows it at once instead of the speed card's spinner, and the next /props
 * answer replaces it if the hardware changed.
 */
function seenHardwareKey(modelId: string): string {
	return `${STORAGE_PREFIX}:session:model:${modelId}:hwSeen`;
}

export function loadSeenHardware(modelId: string): string | null {
	try {
		const v = sessionStorage.getItem(seenHardwareKey(modelId));

		return v && v.trim() ? v : null;
	} catch {
		return null;
	}
}

export function saveSeenHardware(modelId: string, label: string): void {
	try {
		sessionStorage.setItem(seenHardwareKey(modelId), label);
	} catch {
		// a private window or blocked storage: the spinner shows until /props answers
	}
}

export function loadHardwareLabel(modelId: string): string | null {
	const v = readPart<string>(modelId, 'hw');

	return typeof v === 'string' && v.trim() ? v : null;
}

/**
 * The Decode tile's "Text copy boost" (owner, 2026-11-03): show the decode peaks with fast text
 * copies (on) or without (off, the default for anyone who has not set it).
 */
export function loadCopyBoost(modelId: string): boolean {
	return readPart<boolean>(modelId, 'copyBoost') === true;
}

/** Per-viewer UI conveniences (selected tab). */
const UI_KEY = `${STORAGE_PREFIX}:ui`;

export function loadUiState(): { selectedModel?: string; speedView?: 'card' | 'chart' } {
	const v = readJson<{ selectedModel?: string; speedView?: 'card' | 'chart' }>(UI_KEY);

	return isObject(v) ? v : {};
}

export function saveUiState(state: { selectedModel?: string; speedView?: 'card' | 'chart' }): void {
	writeJson(UI_KEY, { ...loadUiState(), ...state });
}
