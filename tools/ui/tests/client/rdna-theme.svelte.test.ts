// RDNA-FAST light and dark themes in a real browser: the tokens follow mode-watcher's `.dark` on
// <html>, a pinned subtree keeps its theme, and the speed card and bench chart (exported as
// images) look the same in both themes.
import RdnaBenchBar from '$lib/components/app/rdna/RdnaBenchBar.svelte';
import RdnaBenchChart from '$lib/components/app/rdna/RdnaBenchChart.svelte';
import RdnaDepthTile from '$lib/components/app/rdna/RdnaDepthTile.svelte';
import RdnaSpeedCard from '$lib/components/app/rdna/RdnaSpeedCard.svelte';
import RdnaWordmark from '$lib/components/app/rdna/RdnaWordmark.svelte';
import { benchRunner } from '$lib/rdna/bench/runner.svelte';
import type { BenchRun } from '$lib/rdna/bench/types';
import { afterAll, afterEach, beforeAll, describe, expect, it } from 'vitest';
import { render } from 'vitest-browser-svelte';

const html = document.documentElement;

let wasDark = false;

beforeAll(() => {
	wasDark = html.classList.contains('dark');
});

afterEach(() => {
	html.classList.remove('dark');
});

afterAll(() => {
	html.classList.toggle('dark', wasDark);
});

function setTheme(theme: 'dark' | 'light'): void {
	html.classList.toggle('dark', theme === 'dark');
}

const css = (el: Element, prop: string) => getComputedStyle(el).getPropertyValue(prop).trim();
const rows = [
	{ hi: 8192, label: 'up to 8k', live: null, peak: 65.8, record: false },
	{ hi: 24576, label: '8k to 24k', live: 61.2, peak: 64, record: true },
	{ hi: 1_000_000_000, label: '24k+', live: null, peak: 61.1, record: false }
];

describe('theme tokens', () => {
	it('switch the tiles between light and dark with the theme setting', async () => {
		setTheme('light');

		const { container } = await render(RdnaDepthTile, { kind: 'decode', onReset: () => {}, rows });
		const tile = container.querySelector('.rf-tile') as HTMLElement;
		const title = container.querySelector('.rf-tile h2') as HTMLElement;
		const line = container.querySelectorAll('.rf-depth-line')[1];
		const live = container.querySelectorAll('.rf-depth-live')[1];

		expect(css(tile, 'background-color')).toBe('rgb(255, 255, 255)');
		expect(css(tile, 'border-top-color')).toBe('rgb(228, 228, 231)');
		expect(css(title, 'color')).toBe('rgb(200, 16, 46)');
		// the live line and its number are dark on white
		expect(css(line, 'background-color')).toBe('rgba(24, 24, 27, 0.6)');
		expect(css(live, 'color')).toBe('rgb(24, 24, 27)');
		// the record glow is red-orange, not white, so it shows on white
		expect(css(tile, '--rf-glow-bar-core')).toBe('rgba(237, 28, 36, 0.6)');

		setTheme('dark');
		expect(css(tile, 'background-color')).toBe('rgb(20, 20, 23)');
		expect(css(tile, 'border-top-color')).toBe('rgb(38, 38, 43)');
		expect(css(title, 'color')).toBe('rgb(255, 81, 72)');
		expect(css(line, 'background-color')).toBe('rgba(255, 255, 255, 0.62)');
		expect(css(live, 'color')).toBe('rgb(255, 255, 255)');
		expect(css(tile, '--rf-glow-bar-core')).toBe('rgba(255, 255, 255, 0.55)');
	});

	it('write the bench progress in the main text colour: white in dark, near-black in light', async () => {
		benchRunner.modelId = 'm';
		benchRunner.stepIndex = 1;
		benchRunner.stepCount = 5;
		benchRunner.stepName = 'code edits';
		benchRunner.phase = 'running';

		try {
			setTheme('dark');

			const { container } = await render(RdnaBenchBar, {
				modelId: 'm',
				onShowResults: () => {},
				view: 'card'
			});
			const label = container.querySelector('.rf-benchbar-label') as HTMLElement;

			expect(label.textContent).not.toBe('');
			expect(css(label, 'color')).toBe('rgb(244, 244, 245)');

			setTheme('light');
			expect(css(label, 'color')).toBe('rgb(24, 24, 27)');
		} finally {
			benchRunner.phase = 'idle';
			benchRunner.modelId = null;
		}
	});

	it('keep a subtree pinned to light inside the dark theme', async () => {
		setTheme('dark');

		const { container } = await render(RdnaDepthTile, { kind: 'prefill', onReset: () => {}, rows });

		container.classList.add('rf-theme-light');
		expect(css(container.querySelector('.rf-tile') as Element, 'background-color')).toBe(
			'rgb(255, 255, 255)'
		);
	});

	it('draw the wordmark in graphite with a dark outline in light, silver in dark', async () => {
		setTheme('light');

		const { container } = await render(RdnaWordmark, {});
		const stop = container.querySelector('stop') as Element;
		const rdna = container.querySelector('tspan') as Element;

		expect(css(stop, 'stop-color')).toBe('rgb(128, 133, 142)');
		expect(css(rdna, 'stroke')).toBe('rgb(24, 24, 27)');

		setTheme('dark');
		expect(css(stop, 'stop-color')).toBe('rgb(255, 255, 255)');
		expect(css(rdna, 'stroke')).toBe('rgb(107, 112, 121)');
	});
});

/** Every colour-related computed style of an element and its descendants. */
function paint(root: Element): string[] {
	const props = [
		'color',
		'background-color',
		'background-image',
		'border-top-color',
		'box-shadow',
		'text-shadow',
		'filter',
		'fill',
		'stroke',
		'stop-color'
	];

	return [root, ...Array.from(root.querySelectorAll('*'))].map((el) =>
		props.map((p) => `${p}=${css(el, p)}`).join(';')
	);
}

function run(t: number, values: BenchRun['values']): BenchRun {
	return {
		build: 'b12781-707407de1',
		code: null,
		edits: [],
		labels: { new_17k: 4338, new_41k: 3993, tok_4k: 4025, tok_17k: 17039, tok_41k: 41053 },
		modelId: 'qwen/qwen3.8-27b',
		modelLabel: 'Swift-1.5-Qwen3.8-27B IQ4_XS',
		nCtx: 98304,
		overlapped: false,
		prose: null,
		sessions: [],
		skipped: [],
		t,
		v: 1,
		values
	};
}

describe('exported images stay dark in both themes', () => {
	it('paints the speed card the same in light and dark', async () => {
		setTheme('light');

		const { container } = await render(RdnaSpeedCard, {
			avatar: null,
			copy: 311,
			decode: 65.8,
			footer: 'Swift-1.5-Qwen3.8-27B IQ4_XS · 96k context · b12781 · bench',
			hardware: 'AMD Radeon RX 7900 XT',
			modelName: 'Swift-1.5-Qwen3.8-27B',
			prefill: 1002
		});
		const card = container.querySelector('.rf-card') as HTMLElement;

		expect(card.classList.contains('rf-theme-dark')).toBe(true);
		expect(css(card, 'color')).toBe('rgb(244, 244, 245)');

		const lightPaint = paint(card);

		setTheme('dark');
		expect(paint(card)).toEqual(lightPaint);
	});

	it('paints the bench chart the same in light and dark', async () => {
		setTheme('light');

		const { container } = await render(RdnaBenchChart, {
			avatar: null,
			hardware: 'AMD Radeon RX 7900 XT',
			modelName: 'Swift-1.5-Qwen3.8-27B',
			now: run(Date.UTC(2026, 10, 3, 9), { decode_4k: 65.8, prefill_4k: 1002, session: 96.7 }),
			prev: run(Date.UTC(2026, 10, 2, 9), { decode_4k: 68.8, prefill_4k: 995, session: 95.9 }),
			quant: 'IQ4_XS'
		});
		const chart = container.querySelector('.rf-chart') as HTMLElement;

		expect(chart.classList.contains('rf-theme-dark')).toBe(true);
		expect(css(chart, 'color')).toBe('rgb(244, 244, 245)');

		const lightPaint = paint(chart);

		setTheme('dark');
		expect(paint(chart)).toEqual(lightPaint);
	});
});
