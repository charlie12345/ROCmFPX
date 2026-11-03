<script lang="ts">
	/**
	 * The big speed tile: the 16:9 speed card or the bench chart, both from finished bench runs only,
	 * the bench's progress bar while a run is active or just finished,
	 * and the bottom bar: view tabs, "Run bench" and the page's only "Save image" (the card as a
	 * 1200 x 675 PNG; the chart 1200 px wide).
	 */
	import RdnaBenchBar from './RdnaBenchBar.svelte';
	import RdnaBenchChart from './RdnaBenchChart.svelte';
	import RdnaSpeedCard from './RdnaSpeedCard.svelte';
	import { benchRunner } from '$lib/rdna/bench/runner.svelte';
	import type { BenchRun } from '$lib/rdna/bench/types';
	import { type CardFigures, cardFooter, cardSource } from '$lib/rdna/card';
	import { pngName, saveNodeAsPng } from '$lib/rdna/export-image';
	import { loadUiState, saveUiState } from '$lib/rdna/storage';
	import { toast } from 'svelte-sonner';

	interface Props {
		card: CardFigures;
		runs: BenchRun[];
		hardware: string | null;
		/** False until the hardware label is settled: the card shows no label and no empty line yet. */
		hardwareKnown?: boolean;
		avatar: string | null;
		/** The model's NAME (model-info.ts): on the card under the figures, the chart's title. */
		modelName: string;
		modelId: string;
		/** NAME and quant, for the saved images' file names. */
		modelLabel: string;
		quant: string | null;
		nCtx: number | null;
		buildInfo: string | null;
		benchDisabled: boolean;
		benchTitle: string;
		onRunBench: () => void;
		onHardwareChange: (label: string | null) => void;
		/** Does a figure glow (record-glow.svelte.ts): `card:prefill`, `chart:decode_4k`, ... */
		isGlowing?: (key: string) => boolean;
	}

	let {
		avatar,
		benchDisabled,
		benchTitle,
		buildInfo,
		card,
		hardware,
		hardwareKnown = true,
		isGlowing = () => false,
		modelId,
		modelLabel,
		modelName,
		nCtx,
		onHardwareChange,
		onRunBench,
		quant,
		runs
	}: Props = $props();

	let view = $state<'card' | 'chart'>(loadUiState().speedView ?? 'card');
	let cardNode = $state<HTMLElement | null>(null);
	let chartNode = $state<HTMLElement | null>(null);
	let saving = $state(false);

	let footer = $derived(
		cardFooter(
			modelLabel,
			runs[0]?.nCtx ?? nCtx,
			runs[0]?.build ?? buildInfo,
			cardSource(runs.length)
		)
	);

	/** Switch the view; a finished run's bar goes with it (one in progress stays). */
	function pick(v: 'card' | 'chart'): void {
		if (v === view) return;

		view = v;
		saveUiState({ speedView: v });

		if (!benchRunner.running) benchRunner.close();
	}

	async function save(): Promise<void> {
		const node = view === 'card' ? cardNode : chartNode;

		if (!node || saving) return;

		saving = true;

		try {
			if (view === 'card') {
				await saveNodeAsPng(node, pngName(modelLabel, 'speed'), { height: 675, width: 1200 });
			} else {
				await saveNodeAsPng(node, pngName(modelLabel, 'bench'), { width: 1200 }, '#0b0b0d');
			}
		} catch (e) {
			toast.error(`Could not save the image: ${e instanceof Error ? e.message : String(e)}`);
		} finally {
			saving = false;
		}
	}
</script>

<section aria-label="Speed" class="rf-speed">
	{#if view === 'card'}
		<RdnaSpeedCard
			bind:node={cardNode}
			{avatar}
			copy={card.copy}
			decode={card.decode}
			editable
			{footer}
			{hardware}
			{hardwareKnown}
			modelKey={modelId}
			{modelName}
			{onHardwareChange}
			prefill={card.prefill}
		/>
	{:else}
		<RdnaBenchChart
			bind:node={chartNode}
			{avatar}
			glowing={(key) => isGlowing(`chart:${key}`)}
			{hardware}
			{modelName}
			now={runs[0] ?? null}
			prev={runs[1] ?? null}
			{quant}
		/>
	{/if}

	{#if benchRunner.visible}
		<RdnaBenchBar {modelId} onShowResults={() => pick('chart')} {view} />
	{/if}

	<div class="rf-speed-bar">
		<div aria-label="Speed view" class="rf-speed-tabs" role="tablist">
			<button
				class:rf-on={view === 'card'}
				aria-selected={view === 'card'}
				onclick={() => pick('card')}
				role="tab"
				type="button">Speed card</button
			>

			<button
				class:rf-on={view === 'chart'}
				aria-selected={view === 'chart'}
				onclick={() => pick('chart')}
				role="tab"
				type="button">Bench chart</button
			>
		</div>

		<div class="rf-speed-actions">
			<button
				class="rf-run"
				disabled={benchDisabled}
				onclick={onRunBench}
				title={benchTitle}
				type="button"
			>
				<svg
					aria-hidden="true"
					fill="none"
					height="16"
					stroke="currentColor"
					stroke-linecap="round"
					stroke-linejoin="round"
					stroke-width="2"
					viewBox="0 0 24 24"
					width="16"><path d="M6 4l14 8-14 8z" /></svg
				>Run bench
			</button>

			<button class="rf-save" disabled={saving} onclick={save} type="button">
				<svg
					aria-hidden="true"
					fill="none"
					height="16"
					stroke="currentColor"
					stroke-linecap="round"
					stroke-linejoin="round"
					stroke-width="2"
					viewBox="0 0 24 24"
					width="16"><path d="M12 4v11M7 10l5 5 5-5M5 20h14" /></svg
				>{saving ? 'Saving…' : 'Save image'}
			</button>
		</div>
	</div>
</section>

<style>
	.rf-speed {
		flex: 3 1 680px;
		min-width: 0;
		border-radius: 16px;
		border: 1px solid var(--rf-border);
		background: var(--rf-panel);
		padding: 12px;
		display: flex;
		flex-direction: column;
		gap: 12px;
		box-sizing: border-box;
	}

	.rf-speed-bar {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		justify-content: space-between;
		gap: 10px;
	}

	.rf-speed-tabs {
		display: inline-flex;
		gap: 4px;
		padding: 4px;
		border-radius: 10px;
		background: var(--rf-panel-deep);
		border: 1px solid var(--rf-border);
	}

	.rf-speed-tabs button {
		padding: 8px 14px;
		border-radius: 7px;
		border: none;
		font-size: 13px;
		font-weight: 700;
		font-family: inherit;
		cursor: pointer;
		background: transparent;
		color: var(--rf-muted);
	}

	.rf-speed-tabs button.rf-on {
		background: var(--rf-tag-bg);
		color: var(--rf-warm-fg);
	}

	.rf-speed-actions {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 8px;
	}

	.rf-run {
		display: inline-flex;
		align-items: center;
		gap: 8px;
		padding: 9px 14px;
		border-radius: 9px;
		border: 1px solid transparent;
		background: var(--rf-grad);
		color: var(--rf-on-grad);
		font-size: 13px;
		font-weight: 700;
		font-family: inherit;
		cursor: pointer;
	}

	.rf-run:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.rf-speed-tabs button:focus-visible,
	.rf-run:focus-visible,
	.rf-save:focus-visible {
		outline: 2px solid var(--rf-orange);
		outline-offset: 2px;
	}

	.rf-save {
		display: inline-flex;
		align-items: center;
		gap: 8px;
		padding: 9px 14px;
		border-radius: 9px;
		border: 1px solid var(--rf-border-strong);
		background: transparent;
		color: var(--rf-text);
		font-size: 13px;
		font-weight: 700;
		font-family: inherit;
		cursor: pointer;
	}

	.rf-save:hover {
		border-color: var(--rf-muted);
	}

	.rf-save:disabled {
		opacity: 0.6;
		cursor: progress;
	}

	@media (max-width: 640px) {
		.rf-speed {
			flex-basis: 100%;
		}
	}
</style>
