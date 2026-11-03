// RDNA-FAST status page in a real browser: Run bench and the bench's progress inside the speed
// tile, and request tables whose columns never move with their content.
import RdnaBenchChart from '$lib/components/app/rdna/RdnaBenchChart.svelte';
import RdnaClassifierTiles from '$lib/components/app/rdna/RdnaClassifierTiles.svelte';
import RdnaRequestsTile from '$lib/components/app/rdna/RdnaRequestsTile.svelte';
import RdnaSpeedTile from '$lib/components/app/rdna/RdnaSpeedTile.svelte';
import { benchRunner } from '$lib/rdna/bench/runner.svelte';
import type { BenchRun } from '$lib/rdna/bench/types';
import type { RunningRow } from '$lib/rdna/status-store.svelte';
import { saveUiState } from '$lib/rdna/storage';
import { emptyClassifier } from '$lib/rdna/storage';
import type { ClassifierRecord, CompletionRecord } from '$lib/rdna/types';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { render } from 'vitest-browser-svelte';

const tile = {
	avatar: null,
	benchDisabled: false,
	benchTitle: 'Run the bench',
	buildInfo: 'b12781-707407de1',
	card: { copy: 311, decode: 65.8, empty: false, prefill: 1002 },
	hardware: 'AMD Radeon RX 7900 XT',
	modelId: 'qwen/qwen3.8-27b',
	modelLabel: 'Swift-1.5-Qwen3.8-27B IQ4_XS',
	modelName: 'Swift-1.5-Qwen3.8-27B',
	nCtx: 98304,
	onHardwareChange: () => {},
	quant: 'IQ4_XS',
	runs: []
};

beforeEach(() => {
	// every test starts on the Speed card view (the view is remembered per browser)
	saveUiState({ speedView: 'card' });
});

afterEach(() => {
	benchRunner.phase = 'idle';
	benchRunner.othersHeld = 0;
	benchRunner.keptIncomplete = false;
});

const tick = () => new Promise((r) => setTimeout(r, 20));

describe('speed tile', () => {
	it('puts Run bench right in front of Save image and calls it', async () => {
		let runs = 0;

		const { container } = await render(RdnaSpeedTile, { ...tile, onRunBench: () => runs++ });
		const buttons = Array.from(container.querySelectorAll('.rf-speed-actions button')).map((b) =>
			b.textContent?.trim()
		);

		expect(buttons).toEqual(['Run bench', 'Save image']);

		(container.querySelector('button.rf-run') as HTMLButtonElement).click();
		expect(runs).toBe(1);
		expect(container.textContent).toContain('no rdna-bench-v1 run yet');
	});

	it('disables Run bench with its reason as the tooltip', async () => {
		const { container } = await render(RdnaSpeedTile, {
			...tile,
			benchDisabled: true,
			benchTitle: 'A bench is running',
			onRunBench: () => {}
		});
		const run = container.querySelector('button.rf-run') as HTMLButtonElement;

		expect(run.disabled).toBe(true);
		expect(run.title).toBe('A bench is running');
	});

	it('shows the bench progress above the buttons only while a run is active or just finished', async () => {
		const { container } = await render(RdnaSpeedTile, { ...tile, onRunBench: () => {} });

		expect(container.querySelector('.rf-benchbar')).toBeNull();

		benchRunner.modelId = tile.modelId;
		benchRunner.stepIndex = 2;
		benchRunner.stepCount = 5;
		benchRunner.stepName = 'code edits';
		benchRunner.othersHeld = 1;
		benchRunner.phase = 'running';
		await new Promise((r) => setTimeout(r, 20));

		const bar = container.querySelector('.rf-benchbar') as HTMLElement;
		const actions = container.querySelector('.rf-speed-bar') as HTMLElement;

		expect(bar.textContent).toContain('step 2 of 5: code edits · 1 other request held');
		expect(bar.textContent).toContain('Stop');
		// directly above the tab and button row, as wide as the tile's content
		expect(bar.nextElementSibling).toBe(actions);
		expect(Math.round(bar.getBoundingClientRect().width)).toBe(
			Math.round(actions.getBoundingClientRect().width)
		);

		benchRunner.phase = 'done';
		await new Promise((r) => setTimeout(r, 20));
		// done, on the Speed card: the button leads to the results
		expect(container.querySelector('.rf-benchbar-btn')?.textContent?.trim()).toBe('Results');
	});

	function finished(phase: 'done' | 'stopped' | 'error', kept = false): void {
		benchRunner.modelId = tile.modelId;
		benchRunner.keptIncomplete = kept;
		benchRunner.phase = phase;
	}

	const button = (c: Element) => c.querySelector('.rf-benchbar-btn') as HTMLButtonElement | null;
	const tab = (c: Element, name: string) =>
		Array.from(c.querySelectorAll('.rf-speed-tabs button')).find(
			(b) => b.textContent?.trim() === name
		) as HTMLButtonElement;

	it('shows the results: done on the Speed card reads "Results", a click opens the chart and closes the bar', async () => {
		const { container } = await render(RdnaSpeedTile, { ...tile, onRunBench: () => {} });

		finished('done');
		await tick();
		expect(button(container)?.textContent?.trim()).toBe('Results');

		button(container)?.click();
		await tick();
		expect(container.querySelector('.rf-chart')).not.toBeNull();
		expect(container.querySelector('.rf-card')).toBeNull();
		expect(container.querySelector('.rf-benchbar')).toBeNull();
		expect(benchRunner.phase).toBe('idle');
	});

	it('reads "Results" for a stopped run that kept an incomplete run, "Close" when nothing was measured', async () => {
		const { container } = await render(RdnaSpeedTile, { ...tile, onRunBench: () => {} });

		finished('stopped', true);
		await tick();
		expect(button(container)?.textContent?.trim()).toBe('Results');

		finished('error', true);
		await tick();
		expect(button(container)?.textContent?.trim()).toBe('Results');

		finished('stopped', false);
		await tick();
		expect(button(container)?.textContent?.trim()).toBe('Close');
	});

	it('keeps "Close" on the Bench chart view', async () => {
		saveUiState({ speedView: 'chart' });

		const { container } = await render(RdnaSpeedTile, { ...tile, onRunBench: () => {} });

		finished('done');
		await tick();
		expect(container.querySelector('.rf-chart')).not.toBeNull();
		expect(button(container)?.textContent?.trim()).toBe('Close');
	});

	it('closes a finished run’s bar on a tab switch either way', async () => {
		const { container } = await render(RdnaSpeedTile, { ...tile, onRunBench: () => {} });

		finished('done');
		await tick();
		tab(container, 'Bench chart').click();
		await tick();
		expect(container.querySelector('.rf-benchbar')).toBeNull();

		finished('stopped', true);
		await tick();
		expect(container.querySelector('.rf-benchbar')).not.toBeNull();
		tab(container, 'Speed card').click();
		await tick();
		expect(container.querySelector('.rf-benchbar')).toBeNull();
	});

	it('leaves the bar alone on a tab switch while a run is in progress', async () => {
		const { container } = await render(RdnaSpeedTile, { ...tile, onRunBench: () => {} });

		benchRunner.modelId = tile.modelId;
		benchRunner.phase = 'running';
		await tick();
		tab(container, 'Bench chart').click();
		await tick();
		expect(container.querySelector('.rf-benchbar')).not.toBeNull();
		expect(button(container)?.textContent?.trim()).toBe('Stop');
		tab(container, 'Speed card').click();
		await tick();
		expect(container.querySelector('.rf-benchbar')).not.toBeNull();
	});
});

function completion(
	id: number,
	status: string,
	over: Partial<CompletionRecord> = {}
): CompletionRecord {
	return {
		draft_accepted: 0,
		draft_n: 0,
		gen_ms: 4000,
		id,
		kind: 'completion',
		n_cache: 17044,
		n_ctx: 23741,
		n_gen: 300,
		n_processed: 6397,
		n_prompt: 23441,
		priority: 'default',
		prompt_ms: 7000,
		status,
		t_arrival_ms: 1_793_000_000_000 - id * 1000,
		t_end_ms: 1_793_000_000_000 - id * 1000 + 12000,
		t_first_ms: 1_793_000_000_000 - id * 1000 + 7100,
		...over
	};
}

const running: RunningRow = {
	cached: 17044,
	ctx: 23441,
	decoded: 412,
	decodeTps: 61.8,
	key: '1:9',
	phase: 'decode',
	prefillTps: null,
	priority: 'xhigh',
	processed: 6397,
	slot: 1,
	t: 1_793_000_000_000
};

/** x positions of the header cells and of each body row's cells. */
function columns(container: HTMLElement): { head: number[]; rows: number[][] } {
	const xs = (cells: NodeListOf<Element>) =>
		Array.from(cells).map((c) => Math.round(c.getBoundingClientRect().left));

	return {
		head: xs(container.querySelectorAll('thead th')),
		rows: Array.from(container.querySelectorAll('tbody tr')).map((tr) =>
			xs(tr.querySelectorAll('td'))
		)
	};
}

async function renderIn(component: never, props: Record<string, unknown>) {
	const host = document.createElement('div');

	host.style.width = '1200px';
	document.body.appendChild(host);

	return render(component, { props, target: host } as never);
}

describe('bench chart names the model by its own name', () => {
	function run(t: number): BenchRun {
		return {
			build: 'b12781-707407de1',
			code: null,
			edits: [],
			labels: {},
			modelId: 'qwen/qwen3.8-27b',
			modelLabel: 'Swift-1.5-Qwen3.8-27B IQ4_XS',
			nCtx: 98304,
			overlapped: false,
			prose: null,
			sessions: [],
			skipped: [],
			t,
			v: 1,
			values: { decode_4k: 65.8 }
		};
	}

	it('titles NAME on hardware and keeps the quant and bench details below', async () => {
		const { container } = await render(RdnaBenchChart, {
			avatar: null,
			hardware: 'AMD Radeon RX 7900 XT',
			modelName: 'Swift-1.5-Qwen3.8-27B',
			now: run(Date.UTC(2026, 10, 3)),
			prev: null,
			quant: 'IQ4_XS'
		});

		expect(container.querySelector('.rf-chart-title')?.textContent).toBe(
			'Swift-1.5-Qwen3.8-27B on AMD Radeon RX 7900 XT'
		);
		expect(container.querySelector('.rf-chart-sub')?.textContent).toBe(
			'IQ4_XS · bench at 96k context, this run against the one before'
		);
	});

	it('leaves the quant out cleanly when the model has none', async () => {
		const { container } = await render(RdnaBenchChart, {
			avatar: null,
			hardware: null,
			modelName: 'my-model',
			now: run(Date.UTC(2026, 10, 3)),
			prev: null,
			quant: null
		});

		expect(container.querySelector('.rf-chart-title')?.textContent).toBe('my-model');
		expect(container.querySelector('.rf-chart-sub')?.textContent).toBe(
			'Bench at 96k context, this run against the one before'
		);
	});
});

describe('request tables keep their columns', () => {
	it('completion table: rows with short and long statuses share x positions, whatever the content', async () => {
		const short = await renderIn(RdnaRequestsTile as never, {
			history: [completion(1, 'done'), completion(2, 'done')],
			missing: false,
			now: 1_793_000_005_000,
			running: []
		});
		const long = await renderIn(RdnaRequestsTile as never, {
			history: [
				completion(1, 'done', { draft_accepted: 990, draft_n: 1000, n_gen: 1000 }),
				completion(2, 'cancelled', { n_cache: 1_234_567, n_processed: 99_999 }),
				completion(3, 'error')
			],
			missing: false,
			now: 1_793_000_005_000,
			running: [running]
		});
		const a = columns(short.container);
		const b = columns(long.container);

		expect(b.head).toEqual(a.head);

		for (const row of [...a.rows, ...b.rows]) expect(row).toEqual(a.head);

		expect(long.container.textContent).toContain('done · fast copy');
		expect(long.container.textContent).toContain('slot 1 · decode');
	});

	it('classifier table: short and long answers and types keep the columns', async () => {
		const rec = (id: number, over: Partial<ClassifierRecord>): ClassifierRecord => ({
			answer: 'yes',
			cached: false,
			id,
			kind: 'classifier',
			latency_ms: 300,
			n_options: 2,
			n_prompt: 110,
			priority: 'default',
			prob: 0.9,
			status: 'done',
			t_arrival_ms: 1_793_000_000_000 - id * 1000,
			t_end_ms: 1_793_000_000_000 - id * 1000 + 300,
			type: 'noul',
			...over
		});
		const props = (history: ClassifierRecord[]) => ({
			classifier: emptyClassifier(),
			history,
			missing: false,
			now: 1_793_000_005_000,
			onResetSpeed: () => {},
			onResetTypes: () => {}
		});
		const short = await renderIn(RdnaClassifierTiles as never, props([rec(1, {}), rec(2, {})]));
		const long = await renderIn(
			RdnaClassifierTiles as never,
			props([
				rec(1, {
					answer: 'a much longer answer label than any column should hold',
					n_options: 21,
					type: 'choice'
				}),
				rec(2, { cached: true, latency_ms: 0.06 }),
				rec(3, { answer: '', status: 'error', type: '' })
			])
		);
		const a = columns(short.container);
		const b = columns(long.container);

		expect(b.head).toEqual(a.head);

		for (const row of [...a.rows, ...b.rows]) expect(row).toEqual(a.head);
	});
});
