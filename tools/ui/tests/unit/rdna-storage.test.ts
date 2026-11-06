import { loadPeaks, noteModelFile, readPart, writePart } from '$lib/rdna/storage';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// the page's parts are kept per model FILE under a router id (owner, 2026-11-07): Flash-Next tried under
// qwen/qwen3.8-27b must not show Swift's bench, and Swift's data must be there again when it comes back
const ID = 'qwen/qwen3.8-27b';
const SWIFT = 'E:\\Models\\ukisai\\Swift-1.5-IQ4_XS\\Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf';
const FNX =
	'E:\\Models\\ukisai\\Swift-1.5-Flash-Next-Q2_0\\Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf';
const store = new Map<string, string>();
const peaks = (v: number) => ({
	decode: {},
	decodeNoCopy: { 98304: { t: 1, v } },
	decodeResetAt: 0,
	prefill: {},
	prefillResetAt: 0
});

let n = 0;

beforeEach(() => {
	store.clear();
	vi.stubGlobal('localStorage', {
		getItem: (k: string) => store.get(k) ?? null,
		removeItem: (k: string) => store.delete(k),
		setItem: (k: string, v: string) => store.set(k, v)
	});
});

afterEach(() => {
	vi.unstubAllGlobals();
});

describe('storage per model file', () => {
	it('keeps nothing for an id whose file is not known yet', () => {
		const id = `unknown-${n++}`;

		writePart(id, 'hw', 'RX 7900 XT');
		expect(store.size).toBe(0);
		expect(readPart(id, 'hw')).toBeNull();
	});

	it('gives another file under the same id its own data', () => {
		const id = `${ID}#${n++}`;

		noteModelFile(id, SWIFT, 'Swift-1.5-Qwen3.8-27B');
		writePart(id, 'peaks', peaks(60));
		expect(noteModelFile(id, FNX, 'Swift-Qwen3.8-Flash-Next-GSQ-RCO')).toBe(true);
		expect(loadPeaks(id).decodeNoCopy).toEqual({});
		writePart(id, 'peaks', peaks(30));
		noteModelFile(id, SWIFT, 'Swift-1.5-Qwen3.8-27B');
		expect(Object.values(loadPeaks(id).decodeNoCopy)[0]?.v).toBe(60);
	});

	it('hands the data under the plain id to the file its bench names, not to another', () => {
		const id = `${ID}#${n++}`;

		// what an older page kept: the plain key, a bench run of Swift
		noteModelFile(id, null, null);
		writePart(id, 'peaks', peaks(60));
		writePart(id, 'bench', [
			{ modelId: id, modelLabel: 'Swift-1.5-Qwen3.8-27B', t: 1, values: {} }
		]);

		noteModelFile(id, FNX, 'Swift-Qwen3.8-Flash-Next-GSQ-RCO');
		expect(readPart(id, 'bench')).toBeNull();
		expect(loadPeaks(id).decodeNoCopy).toEqual({});

		noteModelFile(id, SWIFT, 'Swift-1.5-Qwen3.8-27B');
		expect(Object.values(loadPeaks(id).decodeNoCopy)[0]?.v).toBe(60);
		expect(readPart<unknown[]>(id, 'bench')).toHaveLength(1);
	});

	it('hands data without a bench to the first file that comes up', () => {
		const id = `classifier#${n++}`;

		noteModelFile(id, null, null);
		writePart(id, 'hw', 'RX 7900 XT');
		noteModelFile(
			id,
			'E:\\Models\\startlux\\StartLux-Decision-v1.1-2B-Instant-BF16.gguf',
			'StartLux-Decision-v1.1-2B-Instant'
		);
		expect(readPart(id, 'hw')).toBe('RX 7900 XT');
	});
});
