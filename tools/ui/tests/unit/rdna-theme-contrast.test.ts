// RDNA-FAST light and dark themes: WCAG contrast of the brand tokens (rdna.css) and of the chat's
// light brand tokens (app.css), so text stays readable in both themes.
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const read = (path: string) => readFileSync(new URL(path, import.meta.url), 'utf8');
const rdnaCss = read('../../src/lib/rdna/rdna.css');
const appCss = read('../../src/app.css');

type Tokens = Record<string, string>;

function block(css: string, selector: RegExp): string {
	const m = selector.exec(css);

	if (!m) throw new Error(`no block for ${selector}`);

	return m[1];
}

/** The hex colours of a block, by name without the leading dashes. */
function hexTokens(body: string, prefix: string): Tokens {
	const out: Tokens = {};
	const re = new RegExp(`--${prefix}([a-z0-9-]+):\\s*(#[0-9a-fA-F]{6})\\s*;`, 'g');

	for (const m of body.matchAll(re)) out[m[1]] = m[2];

	return out;
}

function luminance(hex: string): number {
	const ch = (i: number) => {
		const c = parseInt(hex.slice(i, i + 2), 16) / 255;

		return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
	};

	return 0.2126 * ch(1) + 0.7152 * ch(3) + 0.0722 * ch(5);
}

function contrast(a: string, b: string): number {
	const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);

	return (hi + 0.05) / (lo + 0.05);
}

const lightBody = block(rdnaCss, /:root,\s*\.rf-theme-light\s*\{([^}]*)\}/);
const darkBody = block(rdnaCss, /\.dark,\s*\.rf-theme-dark\s*\{([^}]*)\}/);
const light = hexTokens(lightBody, 'rf-');
const dark = hexTokens(darkBody, 'rf-');

function names(body: string): string[] {
	return Array.from(body.matchAll(/--rf-([a-z0-9-]+):/g), (m) => m[1]).sort();
}

/** Text tokens and the grounds they sit on (tiles, the page, tab rows, hovered buttons). */
const TEXT = [
	'text',
	'text-soft',
	'text-dim',
	'muted',
	'faint',
	'label',
	'orange',
	'orange-hover',
	'green',
	'amber',
	'live-num'
];
const GROUNDS = ['panel', 'bg', 'panel-deep', 'track'];

describe.each([
	['light', light],
	['dark', dark]
])('%s theme tokens', (_name, t) => {
	it('has a value for every colour token', () => {
		expect(Object.keys(t).sort()).toEqual(Object.keys(light).sort());
	});

	it.each(GROUNDS)('keeps every text colour at 4.5:1 or more on %s', (ground) => {
		for (const fg of TEXT) {
			expect(contrast(t[fg], t[ground]), `${fg} on ${ground}`).toBeGreaterThanOrEqual(4.5);
		}
	});

	it('keeps tags, the active tab and the bench bar readable', () => {
		expect(contrast(t['tag-fg'], t['tag-bg'])).toBeGreaterThanOrEqual(4.5);
		expect(contrast(t['warm-fg'], t['tag-bg'])).toBeGreaterThanOrEqual(4.5);
		expect(contrast(t['warm-fg'], t['warm-bg'])).toBeGreaterThanOrEqual(4.5);
		// the bench progress text is the main text colour on the bench bar
		expect(contrast(t.text, t['warm-bg'])).toBeGreaterThanOrEqual(4.5);
		expect(contrast(t.text, t['input-bg'])).toBeGreaterThanOrEqual(4.5);
	});

	it('keeps the big red numbers at 3:1 or more (large text)', () => {
		for (const ground of GROUNDS) {
			expect(contrast(t.red, t[ground]), `red on ${ground}`).toBeGreaterThanOrEqual(3);
		}
	});
});

describe('light theme specifics', () => {
	it('gives the dark theme every token but the shared gradient and font', () => {
		const d = names(darkBody);

		expect(names(lightBody).filter((n) => !d.includes(n))).toEqual(['font-display', 'grad']);
		expect(d.filter((n) => !names(lightBody).includes(n))).toEqual([]);
	});

	it('uses the light brand ground, panels, lines and text', () => {
		expect(light).toMatchObject({
			bg: '#f6f6f7',
			border: '#e4e4e7',
			muted: '#52525b',
			panel: '#ffffff',
			text: '#18181b'
		});
	});

	it('keeps dots, low bars and field borders at 3:1 or more on white tiles', () => {
		expect(contrast(light['grey-bar'], light.panel)).toBeGreaterThanOrEqual(3);
		expect(contrast(light['grey-bar'], light.track)).toBeGreaterThanOrEqual(3);
		expect(contrast(light['input-border'], light.panel)).toBeGreaterThanOrEqual(3);
	});

	it('draws the wordmark in graphite with a dark outline', () => {
		expect(contrast(light['wm-silver-stroke'], light.panel)).toBeGreaterThanOrEqual(7);
		expect(contrast(light['wm-silver-2'], light.panel)).toBeGreaterThanOrEqual(4.5);
	});

	it('keeps the brand red and the gradient in both themes', () => {
		expect(light.red).toBe(dark.red);
		expect(rdnaCss).toContain('--rf-grad: linear-gradient(90deg, #ed1c24, #ff6a3d);');
	});
});

describe('chat light brand tokens (app.css)', () => {
	const root = hexTokens(block(appCss, /:root\s*\{([^}]*)\}/), '');

	it('keeps chat text readable', () => {
		expect(contrast(root.foreground, root.background)).toBeGreaterThanOrEqual(4.5);
		expect(contrast(root['muted-foreground'], root.background)).toBeGreaterThanOrEqual(4.5);
		expect(contrast(root['muted-foreground'], root.card)).toBeGreaterThanOrEqual(4.5);
		expect(contrast(root['muted-foreground'], root.muted)).toBeGreaterThanOrEqual(4.5);
		expect(contrast(root['primary-foreground'], root.primary)).toBeGreaterThanOrEqual(4.5);
		expect(contrast(root['sidebar-foreground'], root.sidebar)).toBeGreaterThanOrEqual(4.5);
	});
});
