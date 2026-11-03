/**
 * RDNA-FAST: registers Exo 2 (heavy italic 800 / 900, OFL, static/fonts/exo2) for the wordmark and
 * the big numbers. A <style> with @font-face rules (not FontFace objects) so the image export finds
 * and embeds the font; URLs are made absolute from the app's base, which works under any subpath.
 * No font is fetched from a third party.
 */

import { base } from '$app/paths';

const STYLE_ID = 'rdna-fast-fonts';
const FACES: [weight: number, file: string][] = [
	[800, 'Exo2-ExtraBoldItalic-latin.woff2'],
	[900, 'Exo2-BlackItalic-latin.woff2']
];

export function fontUrl(file: string): string {
	return new URL(`${base}/fonts/exo2/${file}`, document.baseURI).href;
}

export function fontFaceCss(urlOf: (file: string) => string = fontUrl): string {
	return FACES.map(
		([weight, file]) =>
			`@font-face{font-family:'Exo 2';font-style:italic;font-weight:${weight};font-display:swap;src:url('${urlOf(file)}') format('woff2');}`
	).join('\n');
}

export function ensureRdnaFonts(): void {
	if (typeof document === 'undefined' || document.getElementById(STYLE_ID)) return;

	const style = document.createElement('style');

	style.id = STYLE_ID;
	style.textContent = fontFaceCss();
	document.head.appendChild(style);
}

/** The @font-face rules with the fonts inlined as data URLs, for the image export. */
export async function inlineFontFaceCss(): Promise<string> {
	const urls = new Map<string, string>();

	for (const [, file] of FACES) {
		try {
			const res = await fetch(fontUrl(file));
			const blob = await res.blob();
			const data = await new Promise<string>((resolve, reject) => {
				const r = new FileReader();

				r.onload = () => resolve(String(r.result));
				r.onerror = () => reject(r.error);
				r.readAsDataURL(blob);
			});

			urls.set(file, data);
		} catch {
			urls.set(file, fontUrl(file));
		}
	}

	return fontFaceCss((file) => urls.get(file) ?? fontUrl(file));
}
