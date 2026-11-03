// RDNA-FAST status page components in a real browser: the speed card's fast / calm rule and its
// label fit, the depth tile's peak bar, live line and record pulse, and the priority pills.
import RdnaCacheTile from '$lib/components/app/rdna/RdnaCacheTile.svelte';
import RdnaDepthTile from '$lib/components/app/rdna/RdnaDepthTile.svelte';
import RdnaPill from '$lib/components/app/rdna/RdnaPill.svelte';
import RdnaSlotsTile from '$lib/components/app/rdna/RdnaSlotsTile.svelte';
import RdnaSpeedCard from '$lib/components/app/rdna/RdnaSpeedCard.svelte';
import { RDNA_SSD_CACHE_DOCS_URL } from '$lib/rdna/constants';
import { describe, expect, it } from 'vitest';
import { render } from 'vitest-browser-svelte';

const card = {
	avatar: null,
	footer:
		'Swift-1.5-Qwen3.8-27B IQ4_XS · 96k context · b12781 · rdna-bench-v1, best of the last two runs',
	hardware: 'AMD Radeon RX 7900 XT',
	modelName: 'Swift-1.5-Qwen3.8-27B'
};

describe('speed card', () => {
	it('shows the sparking cheetah and glowing numbers for >= 800 t/s prefill and >= 50 t/s decode', async () => {
		const { container } = await render(RdnaSpeedCard, {
			...card,
			copy: 311,
			decode: 65.8,
			prefill: 1002
		});
		const img = container.querySelector('img.rf-card-img') as HTMLImageElement;

		expect(img.src).toContain('cheetah-plate.jpg');
		expect(container.querySelector('.rf-card-glow-big')).not.toBeNull();
		expect(container.textContent).toContain('1002');
		expect(container.textContent).toContain('65.8');
		expect(container.textContent).toContain('Swift-1.5-Qwen3.8-27B');
	});

	it('shows the walking cheetah and flat numbers below the limits', async () => {
		const { container } = await render(RdnaSpeedCard, {
			...card,
			copy: 238,
			decode: 50.6,
			prefill: 408
		});
		const img = container.querySelector('img.rf-card-img') as HTMLImageElement;

		expect(img.src).toContain('cheetah-walk.jpg');
		expect(container.querySelector('.rf-card-glow-big')).toBeNull();
	});

	it('shows no numbers, one line and the snoozing cheetah without a bench result', async () => {
		const { container } = await render(RdnaSpeedCard, {
			...card,
			copy: undefined,
			decode: undefined,
			prefill: undefined
		});

		expect((container.querySelector('img.rf-card-img') as HTMLImageElement).src).toContain(
			'cheetah-sleep.jpg'
		);
		expect(container.querySelector('.rf-card-big')).toBeNull();
		expect(container.querySelector('.rf-card-small')).toBeNull();
		expect(container.querySelector('.rf-card-empty')?.textContent).toBe(
			'Run the bench to fill in this card'
		);
		// the hardware label, corner logo and "with <model>" stay
		expect(container.textContent).toContain('AMD Radeon RX 7900 XT');
		expect(container.querySelector('.rf-corner')).not.toBeNull();
		expect(container.textContent).toContain('Swift-1.5-Qwen3.8-27B');
	});

	it('picks the cheetah: no result = sleep, a slow one = walk, a fast one = sparking', async () => {
		const pick = async (prefill?: number, decode?: number, copy?: number) => {
			const { container } = await render(RdnaSpeedCard, { ...card, copy, decode, prefill });
			const imgs = container.querySelectorAll('img.rf-card-img');

			// the first render shows the picture at once, no reveal
			expect(imgs.length).toBe(1);
			expect(container.querySelector('.rf-card-img-reveal, .rf-card-img-pending')).toBeNull();

			return (imgs[0] as HTMLElement).dataset.cheetah;
		};

		expect(await pick()).toBe('sleep');
		// one value is enough to wake it
		expect(await pick(undefined, undefined, 238)).toBe('walk');
		expect(await pick(408, 50.6, 238)).toBe('walk');
		expect(await pick(1002, 49.9, 311)).toBe('walk');
		expect(await pick(1002, 65.8, 311)).toBe('spark');
	});

	it('reveals the new cheetah over the old one when a bench result changes it', async () => {
		const { container, rerender } = await render(RdnaSpeedCard, {
			...card,
			copy: undefined,
			decode: undefined,
			modelKey: 'm',
			prefill: undefined
		});

		await rerender({ copy: 311, decode: 65.8, modelKey: 'm', prefill: 1002 });

		const shown = () =>
			Array.from(container.querySelectorAll<HTMLElement>('img.rf-card-img')).map(
				(i) => i.dataset.cheetah
			);

		// the old picture stays underneath while the new one comes in
		expect(shown()).toEqual(['sleep', 'spark']);
		await expect
			.poll(() => container.querySelector('.rf-card-img-reveal') !== null, { timeout: 3000 })
			.toBe(true);

		// the numbers do not move with it
		expect(container.querySelector('.rf-card-big')?.getAnimations()).toEqual([]);
		// then the old layer goes
		await expect.poll(shown, { timeout: 3000 }).toEqual(['spark']);
		expect(container.querySelector('.rf-card-img-reveal')).toBeNull();
	});

	it('swaps the cheetah at once on a model switch', async () => {
		const { container, rerender } = await render(RdnaSpeedCard, {
			...card,
			copy: undefined,
			decode: undefined,
			modelKey: 'a',
			prefill: undefined
		});

		await rerender({ copy: 311, decode: 65.8, modelKey: 'b', prefill: 1002 });

		const imgs = container.querySelectorAll<HTMLElement>('img.rf-card-img');

		expect(imgs.length).toBe(1);
		expect(imgs[0].dataset.cheetah).toBe('spark');
		expect(container.querySelector('.rf-card-img-reveal, .rf-card-img-pending')).toBeNull();
	});

	it('shrinks a long hardware label so it stays inside its column', async () => {
		const { container } = await render(RdnaSpeedCard, {
			...card,
			copy: 311,
			decode: 65.8,
			hardware: '2× AMD Radeon RX 7900 XTX + AMD Radeon RX 7800 XT',
			prefill: 1002
		});
		const label = container.querySelector('.rf-card-hw') as HTMLElement;
		const column = container.querySelector('.rf-card-center') as HTMLElement;

		await document.fonts.ready;
		await new Promise((r) => setTimeout(r, 50));

		expect(label.getBoundingClientRect().width).toBeLessThanOrEqual(
			column.getBoundingClientRect().width + 1
		);
	});

	it('lets the hardware label be edited when editable', async () => {
		let saved: string | null = 'unchanged';

		const { container } = await render(RdnaSpeedCard, {
			...card,
			copy: 311,
			decode: 65.8,
			editable: true,
			onHardwareChange: (v: string | null) => (saved = v),
			prefill: 1002
		});
		const button = container.querySelector('button.rf-card-hw') as HTMLButtonElement;

		button.click();
		await new Promise((r) => setTimeout(r, 20));

		const input = container.querySelector('input.rf-card-hw') as HTMLInputElement;

		expect(input).not.toBeNull();
		input.value = 'My rig';
		input.dispatchEvent(new Event('input', { bubbles: true }));
		input.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Enter' }));
		await new Promise((r) => setTimeout(r, 20));

		expect(saved).toBe('My rig');
	});
});

describe('depth tile', () => {
	const rows = [
		{ hi: 8192, label: 'up to 8k', live: null, peak: 65.8, record: false },
		{ hi: 24576, label: '8k to 24k', live: 61.2, peak: 64, record: false },
		{ hi: 1_000_000_000, label: '24k+', live: null, peak: 61.1, record: true }
	];

	it('draws peak bars, the live line with its number, and pulses a record', async () => {
		let resets = 0;

		const { container } = await render(RdnaDepthTile, {
			kind: 'decode',
			onReset: () => resets++,
			rows
		});
		const lives = Array.from(container.querySelectorAll('.rf-depth-live')).map(
			(e) => e.textContent
		);
		const peaks = Array.from(container.querySelectorAll('.rf-depth-value')).map(
			(e) => e.textContent
		);

		expect(peaks).toEqual(['65.8', '64.0', '61.1']);
		// the subtitle is just the unit
		expect(container.querySelector('.rf-tile-sub')?.textContent).toBe('peak t/s');
		// the white line and its number show live values only, and never glow
		expect(lives).toEqual(['', '61.2', '']);
		expect(container.querySelectorAll('.rf-record').length).toBe(0);

		// a new record pulses the bar itself
		const bars = container.querySelectorAll('.rf-depth-peak');

		expect(container.querySelectorAll('.rf-record-bar').length).toBe(1);
		expect(bars[2].classList.contains('rf-record-bar')).toBe(true);

		(
			container.querySelector('button[aria-label="Reset decode peaks"]') as HTMLButtonElement
		).click();
		expect(resets).toBe(1);
	});

	it('has the Text copy boost toggle next to Reset on Decode only, off unless set', async () => {
		const flips: boolean[] = [];
		const { container } = await render(RdnaDepthTile, {
			copyBoost: false,
			kind: 'decode',
			onCopyBoost: (on: boolean) => flips.push(on),
			onReset: () => {},
			rows
		});
		const toggle = Array.from(container.querySelectorAll('button')).find(
			(b) => b.textContent?.trim() === 'Text copy boost'
		) as HTMLButtonElement;

		expect(toggle.getAttribute('aria-pressed')).toBe('false');
		// the same subtle style as Reset, right before it
		expect(toggle.classList.contains('rf-reset')).toBe(true);
		expect(toggle.nextElementSibling?.getAttribute('aria-label')).toBe('Reset decode peaks');
		toggle.click();
		expect(flips).toEqual([true]);

		const on = await render(RdnaDepthTile, {
			copyBoost: true,
			kind: 'decode',
			onCopyBoost: () => {},
			onReset: () => {},
			rows
		});

		expect(on.container.querySelector('button[aria-pressed]')?.getAttribute('aria-pressed')).toBe(
			'true'
		);

		const prefill = await render(RdnaDepthTile, { kind: 'prefill', onReset: () => {}, rows });

		expect(prefill.container.querySelector('button[aria-pressed]')).toBeNull();
	});

	it.each([
		['desktop', 560],
		['phone', 358]
	])(
		'packs the rows close without the live number reaching the row above (%s)',
		async (_name, width) => {
			const { container } = await render(RdnaDepthTile, {
				kind: 'prefill',
				onReset: () => {},
				rows: rows.map((r, i) => ({ ...r, live: 900 + i, peak: 1000 }))
			});

			(container as HTMLElement).style.width = `${width}px`;
			(container as HTMLElement).style.display = 'flex';

			const tracks = Array.from(container.querySelectorAll('.rf-depth-track')).map((e) =>
				e.getBoundingClientRect()
			);
			const lives = Array.from(container.querySelectorAll('.rf-depth-live')).map((e) =>
				e.getBoundingClientRect()
			);

			for (let i = 1; i < tracks.length; i++) {
				// the number sits on its own bar, below the bar of the row above
				expect(lives[i].bottom).toBeLessThanOrEqual(tracks[i].top);
				expect(lives[i].top).toBeGreaterThanOrEqual(tracks[i - 1].bottom);
				// tight rows: bar tops at most 50px apart (they were 60px)
				expect(tracks[i].top - tracks[i - 1].top).toBeLessThanOrEqual(50);
			}
		}
	);
});

describe('priority pill', () => {
	it('styles each priority', async () => {
		for (const p of ['xhigh', 'high', 'default', 'low']) {
			const { container } = await render(RdnaPill, { priority: p });

			expect(container.querySelector(`.rf-pill-${p}`)?.textContent).toBe(p);
		}

		const { container } = await render(RdnaPill, { priority: undefined });

		expect(container.querySelector('.rf-pill-default')?.textContent).toBe('default');
	});
});

describe('SSD cache tile', () => {
	const view = {
		available: true,
		avgRestoreMs: null,
		delta: null,
		disk: { enabled: false },
		enabled: false,
		reused: 0,
		savedMs: 0,
		since: 0
	};

	it('shows only a centered hint when the server runs without --cache-disk', async () => {
		const { container } = await render(RdnaCacheTile, { now: Date.now(), onReset: () => {}, view });
		const off = container.querySelector('.rf-cache-off') as HTMLElement;
		const link = off.querySelector('a') as HTMLAnchorElement;

		expect(container.querySelector('h2')?.textContent).toBe('SSD CACHE');
		expect(off.textContent?.replace(/\s+/g, ' ').trim()).toBe(
			'Enable ssd cache to improve agentic performance significantly'
		);
		expect(link.textContent).toBe('ssd cache');
		expect(link.href).toBe(RDNA_SSD_CACHE_DOCS_URL);
		// no numbers, no bar, no Reset
		expect(container.querySelector('button')).toBeNull();
		expect(container.querySelector('.rf-cache-reused')).toBeNull();
		expect(container.querySelector('.rf-track')).toBeNull();

		const style = getComputedStyle(off);

		expect(style.justifyContent).toBe('center');
		expect(style.alignItems).toBe('center');
		expect(style.textAlign).toBe('center');
	});

	it('keeps the dash view when the server does not report the cache at all', async () => {
		const { container } = await render(RdnaCacheTile, {
			now: Date.now(),
			onReset: () => {},
			view: { ...view, available: false, disk: null }
		});

		expect(container.querySelector('.rf-cache-off')).toBeNull();
		expect(container.textContent).toContain('does not report its SSD cache');
		expect(container.querySelector('button[aria-label="Reset SSD cache figures"]')).not.toBeNull();
	});
});

describe('slots tile', () => {
	const base = { hardware: null, loaded: true, memory: null, pool: null, status: 'loaded' };

	it('keeps a finished request on its slot with its last reading, only the dot turns grey', async () => {
		const endedAt = new Date(2026, 10, 3, 12, 3, 41).getTime();
		const { container } = await render(RdnaSlotsTile, {
			...base,
			lastRequests: {
				1: {
					cached: 17044,
					ctx: 23441,
					decoded: 412,
					endedAt,
					lastRate: 61.8,
					lastRateKind: 'decode',
					phase: 'decode',
					priority: 'xhigh',
					processed: 6397,
					task: 9
				}
			},
			live: [
				{
					cached: 0,
					ctx: 3000,
					decoded: 0,
					decodeTps: null,
					phase: 'idle',
					prefillTps: null,
					processed: 0,
					task: 9
				}
			],
			slots: [{ id: 1, id_task: 9, is_processing: false, n_ctx: 98304, priority: 'default' }]
		});
		const dot = container.querySelector('.rf-slot-dot') as HTMLElement;

		expect(dot.title).toBe('Idle · finished 12:03:41');
		expect(dot.style.background).toContain('--rf-grey-bar');
		expect(container.querySelector('.rf-pill')?.textContent).toBe('xhigh');
		expect(container.querySelector('.rf-slot-rate')?.textContent).toBe('61.8 t/s');
		// idle: the detail line shows the context (the number stays with its unit); no separate
		// token line under the bar
		expect(container.querySelector('.rf-slot-line span')?.textContent).toBe(
			'23,441\u00a0tokens in context'
		);
		expect(container.querySelector('.rf-slot-ctx')).toBeNull();
	});

	it('shows the current reading while a request runs', async () => {
		const { container } = await render(RdnaSlotsTile, {
			...base,
			lastRequests: {},
			live: [
				{
					cached: 0,
					ctx: 5000,
					decoded: 40,
					decodeTps: 61.8,
					phase: 'decode',
					prefillTps: null,
					processed: 0,
					task: 3
				}
			],
			slots: [{ id: 0, id_task: 3, is_processing: true, n_ctx: 98304, priority: 'high' }]
		});

		expect(container.querySelector('.rf-slot-rate')?.textContent).toBe('61.8 t/s');
		expect((container.querySelector('.rf-slot-dot') as HTMLElement).title).toBe('Decode');
	});

	it('puts the device left and the memory figure right, whole even when it wraps', async () => {
		const gib = 1024 ** 3;
		const { container } = await render(RdnaSlotsTile, {
			...base,
			hardware: 'AMD Radeon RX 7900 XT',
			lastRequests: {},
			live: [],
			memory: { kind: 'VRAM', total: 20 * gib, used: 18.7 * gib },
			pool: { total: 98304, used: 10900 },
			slots: []
		});

		(container as HTMLElement).style.display = 'flex';

		const line = container.querySelector('.rf-mem-line') as HTMLElement;
		const hw = container.querySelector('.rf-mem-hw') as HTMLElement;
		const fig = container.querySelector('.rf-mem-figure') as HTMLElement;
		const track = container.querySelector('.rf-mem .rf-track') as HTMLElement;
		// all of an inline element's boxes on one line (Chrome gives a box per child run)
		const oneLine = (el: HTMLElement) =>
			new Set(Array.from(el.getClientRects(), (r) => Math.round(r.top))).size === 1;

		expect(hw.textContent?.trim()).toBe('AMD Radeon RX 7900 XT');
		expect(fig.textContent?.replace(/\s+/g, ' ').trim()).toBe('18.7/20.0 GB');

		// wide (the desktop tile) and as narrow as a phone's: the name left, the figure whole and
		// right (on the same row when both fit, else on the next), the bar under both
		for (const [width, sameRow] of [
			['420px', true],
			['250px', false]
		] as const) {
			(container as HTMLElement).style.width = width;
			const l = line.getBoundingClientRect();
			const h = hw.getBoundingClientRect();
			const f = fig.getBoundingClientRect();

			expect(oneLine(fig)).toBe(true);
			expect(Math.abs(h.left - l.left)).toBeLessThan(1);
			expect(Math.abs(f.right - l.right)).toBeLessThan(1);
			expect(Math.abs(f.top - h.top) < 4).toBe(sameRow);
			expect(track.getBoundingClientRect().top).toBeGreaterThanOrEqual(
				Math.max(h.bottom, f.bottom)
			);
		}

		// the pool figure keeps its value and unit together
		expect(oneLine(container.querySelector('.rf-pool .rf-nowrap') as HTMLElement)).toBe(true);
	});
});
