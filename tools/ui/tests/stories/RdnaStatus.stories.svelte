<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import SidebarNavigation from '$lib/components/app/navigation/SidebarNavigation/SidebarNavigation.svelte';
	import RdnaBenchChart from '$lib/components/app/rdna/RdnaBenchChart.svelte';
	import RdnaCacheTile from '$lib/components/app/rdna/RdnaCacheTile.svelte';
	import RdnaDepthTile from '$lib/components/app/rdna/RdnaDepthTile.svelte';
	import RdnaModelTabs from '$lib/components/app/rdna/RdnaModelTabs.svelte';
	import RdnaSlotsTile from '$lib/components/app/rdna/RdnaSlotsTile.svelte';
	import RdnaSpeedCard from '$lib/components/app/rdna/RdnaSpeedCard.svelte';
	import RdnaSpeedTile from '$lib/components/app/rdna/RdnaSpeedTile.svelte';
	import RdnaWordmark from '$lib/components/app/rdna/RdnaWordmark.svelte';
	import type { BenchRun } from '$lib/rdna/bench/types';
	import type { ModelView } from '$lib/rdna/status-store.svelte';
	import { expect } from 'storybook/test';

	// RDNA-FAST status page: the speed card (fast and calm) and the bench chart
	const { Story } = defineMeta({
		component: RdnaSpeedCard,
		parameters: {
			layout: 'padded'
		},
		title: 'RDNA-FAST/Speed card'
	});

	const card = {
		avatar: null,
		footer:
			'Swift-1.5-Qwen3.8-27B IQ4_XS · 96k context · b12781 · rdna-bench-v1, best of the last two runs',
		hardware: 'AMD Radeon RX 7900 XT',
		modelName: 'Swift-1.5-Qwen3.8-27B'
	};

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

	const now = run(Date.UTC(2026, 10, 3, 9), {
		decode_4k: 65.8,
		decode_17k: 64,
		decode_41k: 61.1,
		edit: 311,
		prefill_4k: 1002,
		prefill_17k: 857,
		prefill_41k: 680,
		session: 96.7
	});
	const prev = run(Date.UTC(2026, 10, 2, 9), {
		decode_4k: 68.8,
		decode_17k: 65.3,
		decode_41k: 65.5,
		edit: 308,
		prefill_4k: 995,
		prefill_17k: 862,
		prefill_41k: 680,
		session: 95.9
	});

	// the Decode tile's rows: a live line on the middle one, a record pulse on the last
	const depthRows = [
		{ hi: 8192, label: 'up to 8k', live: null, peak: 65.8, record: false },
		{ hi: 24576, label: '8k to 24k', live: 61.2, peak: 64, record: false },
		{ hi: 1_000_000_000, label: '24k+', live: null, peak: 61.1, record: true }
	];

	// a model that is loading next to an unloaded one
	function modelView(id: string, status: string, loading: boolean): ModelView {
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

	const bg = (el: Element) => getComputedStyle(el).backgroundColor;
	const fg = (el: Element) => getComputedStyle(el).color;
</script>

<Story
	asChild
	name="Fast"
	play={async ({ canvas }) => {
		await expect(canvas.getByText('1002')).toBeInTheDocument();
		await expect(canvas.getByText('AMD Radeon RX 7900 XT')).toBeInTheDocument();
	}}
>
	<div style="width: 900px; max-width: 100%;">
		<RdnaSpeedCard {...card} copy={311} decode={65.8} prefill={1002} />
	</div>
</Story>

<Story
	asChild
	name="Calm"
	play={async ({ canvas }) => {
		await expect(canvas.getByText('408')).toBeInTheDocument();
	}}
>
	<div style="width: 900px; max-width: 100%;">
		<RdnaSpeedCard {...card} copy={238} decode={50.6} prefill={408} />
	</div>
</Story>

<Story
	asChild
	name="NoBenchRun"
	play={async ({ canvas }) => {
		await expect(canvas.getByText('Run the bench to fill in this card')).toBeInTheDocument();
		await expect(canvas.queryByText('t/s prefill')).not.toBeInTheDocument();
	}}
>
	<div style="width: 900px; max-width: 100%;">
		<RdnaSpeedCard {...card} copy={undefined} decode={undefined} prefill={undefined} />
	</div>
</Story>

<Story
	asChild
	name="BenchChart"
	play={async ({ canvas }) => {
		await expect(canvas.getByText('AGENTIC SESSION')).toBeInTheDocument();
		await expect(canvas.getByText('4.3k new at 17k depth')).toBeInTheDocument();
		await expect(canvas.getByText('96.7 s')).toBeInTheDocument();
		// the model's own name on the hardware; the quant and the bench details below it
		await expect(
			canvas.getByText('Swift-1.5-Qwen3.8-27B on AMD Radeon RX 7900 XT')
		).toBeInTheDocument();
		await expect(
			canvas.getByText('IQ4_XS · bench at 96k context, this run against the one before')
		).toBeInTheDocument();
	}}
>
	<div style="width: 1000px; max-width: 100%;">
		<RdnaBenchChart
			avatar={null}
			hardware="AMD Radeon RX 7900 XT"
			modelName="Swift-1.5-Qwen3.8-27B"
			{now}
			{prev}
			quant="IQ4_XS"
		/>
	</div>
</Story>

<Story
	asChild
	name="SpeedTile"
	play={async ({ canvas }) => {
		const run = canvas.getByRole('button', { name: 'Run bench' });
		const save = canvas.getByRole('button', { name: 'Save image' });

		// Run bench sits right in front of Save image
		await expect(run.nextElementSibling).toBe(save);
		await expect(canvas.getByText(/rdna-bench-v1, best of the last two runs/)).toBeInTheDocument();
	}}
>
	<div style="width: 900px; max-width: 100%;">
		<RdnaSpeedTile
			{...card}
			benchDisabled={false}
			benchTitle="Run the RDNA-FAST bench on this model at priority xhigh"
			buildInfo="b12781-707407de1"
			card={{ copy: 311, decode: 65.8, empty: false, prefill: 1002 }}
			modelId="qwen/qwen3.8-27b"
			modelLabel="Swift-1.5-Qwen3.8-27B IQ4_XS"
			nCtx={98304}
			onHardwareChange={() => {}}
			onRunBench={() => {}}
			quant="IQ4_XS"
			runs={[now, prev]}
		/>
	</div>
</Story>

<Story
	asChild
	name="SidebarStatusItem"
	play={async ({ canvas }) => {
		const item = canvas.getByText('Status');

		await expect(item.closest('a')).toHaveAttribute('href', '#/status');
	}}
>
	<div class="flex-column h-screen w-72 bg-background">
		<SidebarNavigation />
	</div>
</Story>

<Story
	asChild
	name="SsdCacheDisabled"
	play={async ({ canvas }) => {
		await expect(canvas.getByText('ssd cache')).toHaveAttribute(
			'href',
			'https://github.com/serrynaimo/RDNA-FAST#the-ssd-prompt-cache'
		);
		await expect(canvas.queryByRole('button', { name: /Reset/ })).not.toBeInTheDocument();
	}}
>
	<div style="width: 560px; max-width: 100%; display: flex;">
		<RdnaCacheTile
			now={Date.now()}
			onReset={() => {}}
			view={{
				available: true,
				avgRestoreMs: null,
				delta: null,
				disk: { enabled: false },
				enabled: false,
				reused: 0,
				savedMs: 0,
				since: 0
			}}
		/>
	</div>
</Story>

<!-- the light theme (the UI's Light setting); rf-theme-light pins it whatever the setting -->
<Story
	asChild
	name="LightDepthTiles"
	play={async ({ canvasElement }) => {
		const tiles = canvasElement.querySelectorAll('.rf-tile');

		await expect(tiles.length).toBe(2);
		await expect(bg(tiles[0])).toBe('rgb(255, 255, 255)');
		await expect(fg(tiles[0].querySelector('h2') as Element)).toBe('rgb(200, 16, 46)');
		await expect(fg(canvasElement.querySelectorAll('.rf-depth-live')[1])).toBe('rgb(24, 24, 27)');
	}}
>
	<div
		class="rf-theme-light"
		style="display: flex; flex-wrap: wrap; gap: 18px; padding: 18px; background: var(--rf-bg); width: 1000px; max-width: 100%;"
	>
		<RdnaDepthTile kind="decode" onReset={() => {}} rows={depthRows} />

		<RdnaDepthTile
			kind="prefill"
			onReset={() => {}}
			rows={depthRows.map((r) => ({
				...r,
				live: r.live === null ? null : r.live * 15,
				peak: r.peak * 15
			}))}
		/>
	</div>
</Story>

<Story
	asChild
	name="LightSpeedTile"
	play={async ({ canvasElement }) => {
		// the tile follows the light theme, the speed card inside it stays dark (it is the image)
		await expect(bg(canvasElement.querySelector('.rf-speed') as Element)).toBe(
			'rgb(255, 255, 255)'
		);
		await expect(fg(canvasElement.querySelector('.rf-card') as Element)).toBe('rgb(244, 244, 245)');
	}}
>
	<div
		class="rf-theme-light"
		style="padding: 18px; background: var(--rf-bg); width: 936px; max-width: 100%;"
	>
		<RdnaSpeedTile
			{...card}
			benchDisabled={false}
			benchTitle="Run the RDNA-FAST bench on this model at priority xhigh"
			buildInfo="b12781-707407de1"
			card={{ copy: 311, decode: 65.8, empty: false, prefill: 1002 }}
			modelId="qwen/qwen3.8-27b"
			modelLabel="Swift-1.5-Qwen3.8-27B IQ4_XS"
			nCtx={98304}
			onHardwareChange={() => {}}
			onRunBench={() => {}}
			quant="IQ4_XS"
			runs={[now, prev]}
		/>
	</div>
</Story>

<Story
	asChild
	name="LightWordmark"
	play={async ({ canvasElement }) => {
		const rdna = canvasElement.querySelector('tspan') as Element;

		// graphite with a dark outline on the light ground
		await expect(getComputedStyle(rdna).stroke).toBe('rgb(24, 24, 27)');
	}}
>
	<div class="rf-theme-light" style="padding: 18px; background: var(--rf-panel); width: 240px;">
		<RdnaWordmark width={176} />
	</div>
</Story>

<Story
	asChild
	name="DarkDepthTiles"
	play={async ({ canvasElement }) => {
		await expect(bg(canvasElement.querySelector('.rf-tile') as Element)).toBe('rgb(20, 20, 23)');
	}}
>
	<div
		class="rf-theme-dark"
		style="display: flex; padding: 18px; background: var(--rf-bg); width: 560px; max-width: 100%;"
	>
		<RdnaDepthTile kind="decode" onReset={() => {}} rows={depthRows} />
	</div>
</Story>

<!-- a model that is loading: a spinner on its tab, pulsing slot dots, no "loading" text -->
<Story
	asChild
	name="LoadingModel"
	play={async ({ canvasElement }) => {
		await expect(canvasElement.querySelector('[aria-label="Loading"]')).not.toBeNull();
		await expect(canvasElement.querySelectorAll('.rf-slot-dot-loading').length).toBe(2);
		await expect(canvasElement.textContent?.toLowerCase()).not.toContain('loading');
	}}
>
	<div style="display: flex; flex-direction: column; gap: 18px; width: 720px; max-width: 100%;">
		<RdnaModelTabs
			hf={{}}
			models={[
				modelView('Swift-1.5-Qwen3.8-27B', 'loading', true),
				modelView('StartLux-Decision', 'unloaded', false)
			]}
			onSelect={() => {}}
			selectedId="Swift-1.5-Qwen3.8-27B"
		/>

		<div style="display: flex; width: 360px;">
			<RdnaSlotsTile
				hardware="AMD Radeon RX 7900 XT"
				lastRequests={{}}
				live={[]}
				loaded={false}
				loading
				memory={null}
				pool={null}
				slots={[{ id: 0, is_processing: false, n_ctx: 98304, n_prompt_tokens: 10217 }, { id: 1 }]}
				status="loading"
			/>
		</div>
	</div>
</Story>
