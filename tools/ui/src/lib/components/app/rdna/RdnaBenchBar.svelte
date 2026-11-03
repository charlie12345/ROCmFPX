<script lang="ts">
	/**
	 * The bench's progress inside the speed tile, above its tab and button row: step name, no time
	 * estimate, Stop while it runs. Once it finished, the button reads "Results" on the Speed card
	 * view when the run left something to look at (done, or stopped / failed with an incomplete run
	 * kept) and is this model's: it switches the tile to the Bench chart and closes the bar.
	 * Otherwise (on the Bench chart, or nothing measured) it reads "Close". A run of another model
	 * says which one.
	 */
	import { benchRunner } from '$lib/rdna/bench/runner.svelte';

	interface Props {
		/** The model the tile shows. */
		modelId: string;
		/** The speed tile's current view. */
		view: 'card' | 'chart';
		/** Switch the speed tile to the Bench chart. */
		onShowResults: () => void;
	}

	let { modelId, onShowResults, view }: Props = $props();

	let results = $derived(
		!benchRunner.running &&
			view === 'card' &&
			benchRunner.hasResults &&
			benchRunner.modelId === modelId
	);

	function act(): void {
		if (benchRunner.running) {
			benchRunner.stop();

			return;
		}

		if (results) onShowResults();

		benchRunner.close();
	}

	let label = $derived(
		benchRunner.modelId && benchRunner.modelId !== modelId && benchRunner.modelLabel
			? `${benchRunner.modelLabel} · ${benchRunner.label}`
			: benchRunner.label
	);
</script>

<div class="rf-benchbar" role="status">
	<span class="rf-benchbar-label">{label}</span>

	<div
		aria-label="Bench progress"
		aria-valuemax="100"
		aria-valuemin="0"
		aria-valuenow={benchRunner.pct}
		class="rf-benchbar-track"
		role="progressbar"
	>
		<div class="rf-benchbar-fill" style="width: {benchRunner.pct}%;"></div>
	</div>

	<button class="rf-benchbar-btn" onclick={act} type="button">
		{benchRunner.running ? 'Stop' : results ? 'Results' : 'Close'}
	</button>
</div>

<style>
	.rf-benchbar {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 8px 14px;
		padding: 10px 12px;
		border-radius: 10px;
		background: var(--rf-warm-bg);
		border: 1px solid var(--rf-warm-border);
	}

	.rf-benchbar-label {
		flex: 1 1 100%;
		font-size: 13px;
		font-weight: 700;
		/* the theme's main text: white in dark, near-black in light */
		color: var(--rf-text);
	}

	.rf-benchbar-track {
		flex: 1 1 200px;
		height: 6px;
		border-radius: 3px;
		background: var(--rf-warm-track);
	}

	.rf-benchbar-fill {
		height: 6px;
		border-radius: 3px;
		background: var(--rf-grad);
		transition: width 0.6s ease;
	}

	.rf-benchbar-btn {
		padding: 6px 12px;
		border-radius: 8px;
		border: 1px solid var(--rf-warm-border-strong);
		background: transparent;
		color: var(--rf-text);
		font-size: 13px;
		font-weight: 600;
		font-family: inherit;
		cursor: pointer;
	}

	.rf-benchbar-btn:hover {
		background: var(--rf-warm-track);
	}

	.rf-benchbar-btn:focus-visible {
		outline: 2px solid var(--rf-orange);
		outline-offset: 2px;
	}
</style>
