// RDNA-FAST status page in a real browser: the tiles' entrance runs once, in reading order, and
// is gone when it ends; the speed card (the saved image) never carries it, and saving waits for it.
import '$lib/rdna/rdna.css';
import RdnaSpeedTile from '$lib/components/app/rdna/RdnaSpeedTile.svelte';
import { ENTER_CLASS, ENTER_STAGGER_MS, entranceMs, playEntrance } from '$lib/rdna/entrance';
import { settled } from '$lib/rdna/export-image';
import { describe, expect, it } from 'vitest';
import { render } from 'vitest-browser-svelte';

function grid(): HTMLElement {
	const root = document.createElement('div');

	root.style.cssText =
		'display: grid; grid-template-columns: 200px 200px; gap: 10px; width: 420px;';

	// DOM order: b, a, d, c, placed so reading order is a, b, c, d
	for (const [name, col, row] of [
		['b', 2, 1],
		['a', 1, 1],
		['d', 2, 2],
		['c', 1, 2]
	] as const) {
		const el = document.createElement('section');

		el.className = 'rf-tile';
		el.dataset.name = name;
		el.style.cssText = `grid-column: ${col}; grid-row: ${row}; height: 60px; border: 1px solid;`;
		root.appendChild(el);
	}

	document.body.appendChild(root);

	return root;
}

describe('tile entrance', () => {
	it('staggers the tiles in reading order and takes its classes off when done', async () => {
		const root = grid();
		const end = playEntrance(root);
		const tiles = Array.from(root.querySelectorAll<HTMLElement>('.rf-tile'));
		const delays = Object.fromEntries(
			tiles.map((t) => [t.dataset.name, t.style.getPropertyValue('--rf-enter-delay')])
		);

		expect(tiles.every((t) => t.classList.contains(ENTER_CLASS))).toBe(true);
		expect(delays).toEqual({
			a: '0ms',
			b: `${ENTER_STAGGER_MS}ms`,
			c: `${2 * ENTER_STAGGER_MS}ms`,
			d: `${3 * ENTER_STAGGER_MS}ms`
		});
		expect(getComputedStyle(tiles[0]).animationName).toContain('rfEnter');
		// the whole page settles in about 1.2 s
		expect(entranceMs(9)).toBeLessThanOrEqual(1500);

		await new Promise((r) => setTimeout(r, entranceMs(4) + 150));
		expect(tiles.some((t) => t.classList.contains(ENTER_CLASS))).toBe(false);
		expect(tiles.some((t) => t.style.getPropertyValue('--rf-enter-delay'))).toBe(false);
		end();
		root.remove();
	});

	it('keeps the speed card out of it, and saving waits until the tile has landed', async () => {
		const { container } = await render(RdnaSpeedTile, {
			avatar: null,
			benchDisabled: false,
			benchTitle: 'Run the bench',
			buildInfo: null,
			card: { copy: 311, decode: 65.8, empty: false, prefill: 1002 },
			hardware: 'AMD Radeon RX 7900 XT',
			modelId: 'm',
			modelLabel: 'Swift-1.5-Qwen3.8-27B IQ4_XS',
			modelName: 'Swift-1.5-Qwen3.8-27B',
			nCtx: 98304,
			onHardwareChange: () => {},
			onRunBench: () => {},
			quant: 'IQ4_XS',
			runs: []
		});
		const end = playEntrance(container as HTMLElement);
		const tile = container.querySelector('.rf-speed') as HTMLElement;
		const card = container.querySelector('.rf-card') as HTMLElement;

		expect(tile.classList.contains(ENTER_CLASS)).toBe(true);
		expect(card.classList.contains(ENTER_CLASS)).toBe(false);
		expect(card.closest('.rf-enter')).toBe(tile);

		await settled(card);
		// nothing finite is still running on the tile once the export may go ahead
		expect(tile.getAnimations().filter((a) => a.playState === 'running')).toEqual([]);
		end();
	});
});
