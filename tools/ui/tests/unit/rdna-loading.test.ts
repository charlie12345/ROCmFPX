// RDNA-FAST status page: a model that is loading (router status "loading", or listed loaded while
// its /slots still answers 503) is marked loading, shows its slots (last known, idle, or empty rows)
// and goes back to normal once its endpoints answer.
import { slotCountOf } from '$lib/rdna/model-info';
import { StatusStore } from '$lib/rdna/status-store.svelte';
import type { RdnaModelEntry, RdnaSlot } from '$lib/rdna/types';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

type Internals = {
	pollSlots(): Promise<void>;
	rootKnown: boolean;
	routerMode: boolean;
};

let slotsAnswer: { body: unknown; status: number } = { body: [], status: 200 };

beforeEach(() => {
	const mem = new Map<string, string>();

	vi.stubGlobal('localStorage', {
		getItem: (k: string) => mem.get(k) ?? null,
		removeItem: (k: string) => mem.delete(k),
		setItem: (k: string, v: string) => mem.set(k, v)
	});
	vi.stubGlobal('fetch', async (url: string) => {
		if (!String(url).includes('/slots')) return new Response('{}', { status: 404 });

		return new Response(JSON.stringify(slotsAnswer.body), { status: slotsAnswer.status });
	});
});

afterEach(() => {
	vi.unstubAllGlobals();
});

function entry(status: string, args: string[] = ['--parallel', '2']): RdnaModelEntry {
	return { id: 'm', meta: { n_ctx: 98304 }, status: { args, value: status } } as RdnaModelEntry;
}

function makeStore(status: string, args?: string[]): StatusStore {
	const store = new StatusStore();
	const s = store as unknown as Internals;

	s.rootKnown = true;
	s.routerMode = true;
	store.entries = [entry(status, args)];
	store.select('m');

	return store;
}

const busySlots: RdnaSlot[] = [
	{ id: 0, id_task: 4, is_processing: true, n_ctx: 98304, n_prompt_tokens: 10217 },
	{ id: 1, id_task: 5, is_processing: false, n_ctx: 98304, n_prompt_tokens: 648 }
];

describe('slot count from the command line', () => {
	it('reads -np / --parallel and ignores nonsense', () => {
		expect(slotCountOf(['-np', '2'])).toBe(2);
		expect(slotCountOf(['--parallel=4'])).toBe(4);
		expect(slotCountOf(['--parallel', 'x'])).toBeNull();
		expect(slotCountOf([])).toBeNull();
	});
});

describe('a loading model', () => {
	it('is loading (not loaded) while the router says so, with one empty row per slot', () => {
		const store = makeStore('loading');

		expect(store.selected).toMatchObject({ loaded: false, loading: true });
		expect(store.shownSlots?.map((s) => s.id)).toEqual([0, 1]);
		expect(store.shownSlots?.every((s) => s.n_prompt_tokens === undefined)).toBe(true);
	});

	it('is loading while its /slots answers 503, then normal once it answers', async () => {
		const store = makeStore('loaded');
		const s = store as unknown as Internals;

		slotsAnswer = { body: { error: { code: 503, message: 'Loading model' } }, status: 503 };
		await s.pollSlots();
		expect(store.selected).toMatchObject({ loaded: true, loading: true });
		expect(store.shownSlots?.length).toBe(2);

		slotsAnswer = { body: busySlots, status: 200 };
		await s.pollSlots();
		expect(store.selected?.loading).toBe(false);
		expect(store.shownSlots).toEqual(busySlots);
	});

	it('shows its last known slots, idle, when it loads again', async () => {
		const store = makeStore('loaded');
		const s = store as unknown as Internals;

		slotsAnswer = { body: busySlots, status: 200 };
		await s.pollSlots();

		// the router unloads and reloads it
		store.entries = [entry('loading')];
		expect(store.selected?.loading).toBe(true);
		expect(store.shownSlots?.map((x) => [x.id, x.n_prompt_tokens, x.is_processing])).toEqual([
			[0, 10217, false],
			[1, 648, false]
		]);
	});

	it('is not loading once the router gives up on it (unloaded keeps the greyed tab)', async () => {
		const store = makeStore('loaded');
		const s = store as unknown as Internals;

		slotsAnswer = { body: {}, status: 503 };
		await s.pollSlots();
		expect(store.selected?.loading).toBe(true);

		store.entries = [entry('unloaded')];
		expect(store.selected).toMatchObject({ loaded: false, loading: false });
	});

	it('falls back to one empty row without -np or /props', () => {
		const store = makeStore('loading', []);

		expect(store.shownSlots).toEqual([{ id: 0 }]);
	});
});
