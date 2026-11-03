// RDNA-FAST status page in a real browser: a loading model has a spinner on its tab and pulsing
// slot dots, and the word "loading" appears nowhere; an unloaded model keeps its greyed tab.
import RdnaModelHeader from '$lib/components/app/rdna/RdnaModelHeader.svelte';
import RdnaModelTabs from '$lib/components/app/rdna/RdnaModelTabs.svelte';
import RdnaSlotsTile from '$lib/components/app/rdna/RdnaSlotsTile.svelte';
import type { ModelView } from '$lib/rdna/status-store.svelte';
import { describe, expect, it } from 'vitest';
import { render } from 'vitest-browser-svelte';

function view(id: string, status: string, loading: boolean): ModelView {
	return {
		args: [],
		classifier: false,
		entry: { id, status: { value: status } },
		id,
		label: id,
		loaded: status === 'loaded',
		loading,
		naming: null,
		runtime: 'ROCm',
		status
	} as ModelView;
}

describe('model tabs', () => {
	it('spin for a loading model, grey an unloaded one, and never say "loading"', async () => {
		const { container } = await render(RdnaModelTabs, {
			hf: {},
			models: [view('swift', 'loading', true), view('other', 'unloaded', false)],
			onSelect: () => {},
			selectedId: 'swift'
		});
		const [loading, unloaded] = Array.from(container.querySelectorAll('.rf-tab')) as HTMLElement[];
		const spinner = loading.querySelector('.rf-tab-spinner') as HTMLElement;

		expect(spinner.getAttribute('aria-label')).toBe('Loading');
		expect(getComputedStyle(spinner).animationName).not.toBe('none');
		// the spinner takes the avatar's place
		expect(loading.querySelector('.rf-avatar')).toBeNull();
		expect(loading.classList.contains('rf-tab-off')).toBe(false);
		expect(loading.title).toBe('swift');
		expect(loading.textContent?.toLowerCase()).not.toContain('loading');

		// unloaded (not loading): greyed with its state, as before
		expect(unloaded.classList.contains('rf-tab-off')).toBe(true);
		expect(unloaded.querySelector('.rf-tab-spinner')).toBeNull();
		expect(unloaded.textContent).toContain('unloaded');
	});

	it('show no state text in the model header while loading', async () => {
		// a component prop named "props" needs the options form of render()
		const { container } = await render(RdnaModelHeader, {
			props: {
				hf: null,
				model: view('swift', 'loading', true),
				nCtx: 98304,
				props: null,
				totalSlots: 2
			}
		});

		expect(container.querySelector('.rf-model-state')).toBeNull();
		expect(container.textContent?.toLowerCase()).not.toContain('loading');
	});
});

describe('slots tile while the model loads', () => {
	const base = { hardware: null, lastRequests: {}, live: [], memory: null, pool: null };

	it('pulses the dots of the slots it will have, with no note', async () => {
		const { container } = await render(RdnaSlotsTile, {
			...base,
			loaded: false,
			loading: true,
			slots: [{ id: 0 }, { id: 1 }],
			status: 'loading'
		});
		const dots = Array.from(container.querySelectorAll('.rf-slot-dot')) as HTMLElement[];

		expect(dots.length).toBe(2);

		for (const dot of dots) {
			expect(dot.classList.contains('rf-slot-dot-loading')).toBe(true);
			// the colour comes from the pulse, not from an inline state colour
			expect(dot.getAttribute('style') ?? '').not.toContain('background');
			expect(getComputedStyle(dot).animationName).not.toBe('none');
		}

		expect(container.querySelector('.rf-note')).toBeNull();
		expect(container.textContent?.toLowerCase()).not.toContain('loading');
		expect(container.querySelector('.rf-slot-line span')?.textContent).toBe('–');
	});

	it('keeps the last known context of a slot while it loads again', async () => {
		const { container } = await render(RdnaSlotsTile, {
			...base,
			loaded: true,
			loading: true,
			slots: [{ id: 0, is_processing: false, n_ctx: 98304, n_prompt_tokens: 10217 }],
			status: 'loaded'
		});

		expect(container.querySelector('.rf-slot-line span')?.textContent).toBe(
			'10,217 tokens in context'
		);
		expect(container.querySelector('.rf-slot-dot-loading')).not.toBeNull();
	});

	it('stops pulsing once the model is loaded', async () => {
		const { container } = await render(RdnaSlotsTile, {
			...base,
			loaded: true,
			loading: false,
			slots: [{ id: 0, is_processing: false, n_ctx: 98304, n_prompt_tokens: 10217 }],
			status: 'loaded'
		});

		expect(container.querySelector('.rf-slot-dot-loading')).toBeNull();
	});
});
