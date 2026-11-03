/**
 * RDNA-FAST: save a part of the page as a PNG, client-side (modern-screenshot renders the node
 * through an SVG foreignObject). The speed card is drawn at exactly 1200 x 675: the node is
 * rendered at the scale that makes it 1200 px wide (everything on the card is sized in container
 * units, so this is the 1200 px layout) and then fitted onto a 1200 x 675 canvas.
 */

import { inlineFontFaceCss } from './fonts';

async function renderCanvas(
	node: HTMLElement,
	scale: number,
	background: string
): Promise<HTMLCanvasElement> {
	const { domToCanvas } = await import('modern-screenshot');

	return domToCanvas(node, {
		backgroundColor: background,
		font: { cssText: await inlineFontFaceCss() },
		scale
	});
}

function download(canvas: HTMLCanvasElement, fileName: string): Promise<void> {
	return new Promise((resolve, reject) => {
		canvas.toBlob((blob) => {
			if (!blob) {
				reject(new Error('the browser could not encode the image'));

				return;
			}

			const url = URL.createObjectURL(blob);
			const a = document.createElement('a');

			a.href = url;
			a.download = fileName;
			document.body.appendChild(a);
			a.click();
			a.remove();
			setTimeout(() => URL.revokeObjectURL(url), 2000);
			resolve();
		}, 'image/png');
	});
}

/**
 * Wait until nothing on or around the node is mid-animation: the page's tile entrance on its
 * ancestors and one-off transitions inside it (the cheetah's reveal). Endless animations (a record
 * glow) are left alone.
 */
export async function settled(node: HTMLElement): Promise<void> {
	if (typeof node.getAnimations !== 'function') return;

	// a picture still loading starts its reveal once it has: let it, then wait for that too
	await Promise.allSettled(Array.from(node.querySelectorAll('img'), (img) => img.decode()));
	await new Promise((r) => {
		requestAnimationFrame(() => r(null));
		setTimeout(() => r(null), 50);
	});

	const anims = [...node.getAnimations({ subtree: true })];

	for (let el = node.parentElement; el; el = el.parentElement) anims.push(...el.getAnimations());

	const finite = anims.filter((a) => a.effect?.getComputedTiming().iterations !== Infinity);

	await Promise.allSettled(finite.map((a) => a.finished));
}

/** Save a node as a PNG of the given size (the node keeps its aspect ratio). */
export async function saveNodeAsPng(
	node: HTMLElement,
	fileName: string,
	size: { width: number; height?: number },
	background = '#000000'
): Promise<void> {
	await document.fonts?.ready;
	// never a tile or an image caught mid-animation
	await settled(node);

	const rect = node.getBoundingClientRect();
	const scale = size.width / Math.max(1, rect.width);
	const drawn = await renderCanvas(node, scale, background);
	const height = size.height ?? Math.round(rect.height * scale);
	const out = document.createElement('canvas');

	out.width = size.width;
	out.height = height;

	const ctx = out.getContext('2d');

	if (!ctx) throw new Error('no 2D canvas');

	ctx.fillStyle = background;
	ctx.fillRect(0, 0, out.width, out.height);
	ctx.drawImage(drawn, 0, 0, out.width, out.height);
	await download(out, fileName);
}

/** A file name from a label: "rdna-fast-swift-1.5-qwen3.8-27b-2026-11-03.png". */
export function pngName(label: string, kind: string, now: Date = new Date()): string {
	const slug = label
		.toLowerCase()
		.replace(/[^a-z0-9.]+/g, '-')
		.replace(/^-+|-+$/g, '')
		.slice(0, 60);
	const day = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;

	return `rdna-fast-${kind}-${slug || 'model'}-${day}.png`;
}
