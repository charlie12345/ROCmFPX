/**
 * RDNA-FAST status page: the tiles' entrance (owner, 2026-11-03), once per page view, when the
 * page first shows its tiles (the models are known). Tiles power on staggered in reading order (row
 * by row, left to right): scanned in from the top behind a glowing scanline with a hologram
 * flicker, HUD corner brackets locking on, then a red border flash as it settles. The motion is CSS
 * (`rf-enter` in rdna.css); this file only orders the tiles, sets each one's delay and takes the
 * class off again when the last one is done. Polling and data are never held up by it. The new
 * chat page plays the same entrance on its prompt input box, once each time the page opens (owner,
 * 2026-11-04).
 */

/** The status page's tiles (the model header, Speed, Slots, every RdnaTile, the request lists). */
export const ENTRANCE_TILES = '.rf-model, .rf-speed, .rf-slots, .rf-tile, .rf-req, .rf-cls-req';

/** The chat page's prompt input box. */
export const ENTRANCE_CHAT_INPUT = 'form.conversation-chat-form [data-slot="input-area"]';

/** Keep these in step with `.rf-enter` in rdna.css. */
export const ENTER_STAGGER_MS = 70;
export const ENTER_MS = 480;
/** The brackets and the border flash end this long after the tile starts (the lock-in). */
export const ENTER_LANDING_END_MS = 880;

export const ENTER_CLASS = 'rf-enter';

/** How long the entrance of `n` tiles takes, the last one's landing included. */
export function entranceMs(n: number): number {
	return Math.max(0, n - 1) * ENTER_STAGGER_MS + Math.max(ENTER_MS, ENTER_LANDING_END_MS);
}

/** Elements in reading order: rows by their top edge (8 px tolerance), each row left to right. */
export function readingOrder(els: readonly HTMLElement[]): HTMLElement[] {
	const placed = els
		.map((el) => ({ el, r: el.getBoundingClientRect() }))
		.sort((a, b) => a.r.top - b.r.top);
	const rows: (typeof placed)[] = [];

	for (const p of placed) {
		const row = rows[rows.length - 1];

		if (row && Math.abs(p.r.top - row[0].r.top) <= 8) row.push(p);
		else rows.push([p]);
	}

	return rows.flatMap((row) => row.sort((a, b) => a.r.left - b.r.left).map((p) => p.el));
}

/**
 * Play the entrance on the tiles under `root`. Returns a function that ends it at once (the page
 * calls it when it goes away).
 */
export function playEntrance(root: HTMLElement, selector = ENTRANCE_TILES): () => void {
	const all = Array.from(root.querySelectorAll<HTMLElement>(selector));
	const tiles = readingOrder(all.filter((el) => !all.some((o) => o !== el && o.contains(el))));

	tiles.forEach((el, i) => {
		el.style.setProperty('--rf-enter-delay', `${i * ENTER_STAGGER_MS}ms`);
		el.classList.add(ENTER_CLASS);
	});

	let ended = false;

	const end = () => {
		if (ended) return;

		ended = true;
		clearTimeout(timer);

		for (const el of tiles) {
			el.classList.remove(ENTER_CLASS);
			el.style.removeProperty('--rf-enter-delay');
		}
	};
	const timer = setTimeout(end, entranceMs(tiles.length) + 50);

	return end;
}
