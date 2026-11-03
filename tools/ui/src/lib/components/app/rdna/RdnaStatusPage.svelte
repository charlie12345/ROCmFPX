<script lang="ts">
	/**
	 * RDNA-FAST status page (Main.dc.html, Classifier.dc.html) in the chat's sidebar layout: model
	 * tabs, model header and the tiles. A completion model shows Speed (with Run bench and the bench's
	 * progress) + Slots, Decode + Prefill, SSD cache + Priority and Requests; a classifier Speed,
	 * Question types and its request list.
	 */
	import '$lib/rdna/rdna.css';
	import RdnaCacheTile from './RdnaCacheTile.svelte';
	import RdnaClassifierTiles from './RdnaClassifierTiles.svelte';
	import RdnaDepthTile from './RdnaDepthTile.svelte';
	import RdnaModelHeader from './RdnaModelHeader.svelte';
	import RdnaModelTabs from './RdnaModelTabs.svelte';
	import RdnaPriorityTile from './RdnaPriorityTile.svelte';
	import RdnaRequestsTile from './RdnaRequestsTile.svelte';
	import RdnaSlotsTile from './RdnaSlotsTile.svelte';
	import RdnaSpeedTile from './RdnaSpeedTile.svelte';
	import { ServerErrorSplash } from '$lib/components/app';
	import { benchRunner } from '$lib/rdna/bench/runner.svelte';
	import { playEntrance } from '$lib/rdna/entrance';
	import { ensureRdnaFonts } from '$lib/rdna/fonts';
	import { StatusStore } from '$lib/rdna/status-store.svelte';
	import { onDestroy, onMount } from 'svelte';

	const store = new StatusStore();

	onMount(() => {
		ensureRdnaFonts();
		store.start();
	});

	let mainEl = $state<HTMLElement | null>(null);
	let entered = false;
	let endEntrance: (() => void) | null = null;

	onDestroy(() => {
		store.stop();
		endEntrance?.();
	});

	// a run saved by the bench (it may finish while another page is open)
	$effect(() => {
		void benchRunner.savedAt;
		store.reloadBench();
	});

	// the running bench fills the speed card and the chart in: improvements glow
	$effect(() => {
		void benchRunner.partial;
		store.observeRecords();
	});

	let model = $derived(store.selected);

	// the tiles' entrance, once per page view: when the tiles first show (the models are known),
	// never again on tab or view switches or data updates (entrance.ts)
	$effect(() => {
		if (entered || !mainEl || !model || store.auth === 'needed') return;

		entered = true;
		endEntrance = playEntrance(mainEl);
	});
	let modelLabel = $derived(
		model ? [model.naming?.name ?? model.id, model.naming?.quant].filter(Boolean).join(' ') : ''
	);
	// the model's own NAME, as the header and the tabs show it (never the HF base model)
	let modelName = $derived(model?.naming?.name || model?.label || '');

	let benchBlock = $derived.by(() => {
		if (benchRunner.running) return 'A bench is running';

		if (!model) return 'No model';

		if (model.loading) return 'The model is starting: the bench runs once it is ready';

		if (!model.loaded) return 'Load the model first: the bench runs on a loaded model';

		if (model.classifier) return 'The bench measures chat models; this one is a classifier';

		if (store.auth === 'needed') return 'Enter the API key first';

		return null;
	});

	function runBench(): void {
		if (benchBlock || !model) return;

		void benchRunner.start({
			build: store.buildInfo,
			modelId: model.id,
			modelLabel,
			nCtx: store.slotCtx,
			routerMode: store.routerMode
		});
	}
</script>

<svelte:head>
	<title>RDNA-FAST status{model ? ` · ${model.label}` : ''}</title>
</svelte:head>

<!-- no or a refused API key: the chat's own splash (owner, 2026-11-03); a valid key stays here -->
{#if store.auth === 'needed'}
	<ServerErrorSplash
		error="Access denied"
		onRetry={() => store.retryAuth()}
		showRetry={false}
		showTroubleshooting={false}
	/>
{:else}
	<div class="rf-page">
		<main bind:this={mainEl} class="rf-main">
			{#if store.online === false && !store.modelsLoaded}
				<p class="rf-empty">The server does not answer. The page keeps trying every few seconds.</p>
			{:else if !store.modelsLoaded}
				<p class="rf-empty">Connecting…</p>
			{:else if store.models.length === 0}
				<p class="rf-empty">The server lists no models.</p>
			{:else}
				<RdnaModelTabs
					hf={store.hf}
					models={store.models}
					onSelect={(id) => store.select(id)}
					selectedId={store.selectedId}
				/>

				{#if model}
					<RdnaModelHeader
						hf={store.hfView}
						{model}
						nCtx={store.nCtx}
						props={store.selectedProps}
						totalSlots={store.totalSlots}
					/>

					<div class:rf-grid-top={!model.classifier} class="rf-grid">
						{#if model.classifier}
							<RdnaClassifierTiles
								classifier={store.classifier}
								glowFastest={store.glow.on('cls:fastest', store.now)}
								glowPrompt={store.glow.on('cls:prompt', store.now)}
								history={store.history}
								missing={!!store.requestsMissing[model.id]}
								now={store.now}
								onResetSpeed={() => store.resetClassifierSpeed()}
								onResetTypes={() => store.resetQuestionTypes()}
							/>
						{:else}
							<RdnaSpeedTile
								avatar={store.hfView?.avatar ?? null}
								benchDisabled={benchBlock !== null}
								benchTitle={benchBlock ??
									'Run the RDNA-FAST bench on this model at priority xhigh (other requests are held while it runs)'}
								buildInfo={store.buildInfo}
								card={store.card}
								hardware={store.hardware}
								hardwareKnown={store.hardwareKnown}
								isGlowing={(key) => store.glow.on(key, store.now)}
								modelId={model.id}
								{modelLabel}
								{modelName}
								nCtx={store.nCtx}
								onHardwareChange={(label) => store.setHardwareLabel(label)}
								onRunBench={runBench}
								quant={model.naming?.quant ?? null}
								runs={store.runsNewestFirst}
							/>

							<RdnaSlotsTile
								hardware={store.hardware}
								lastRequests={store.slotMemory}
								live={store.live}
								loaded={model.loaded}
								loading={model.loading}
								memory={store.memory}
								pool={store.sharedPool}
								slots={store.shownSlots}
								status={model.status}
							/>

							<RdnaDepthTile
								copyBoost={store.copyBoost}
								kind="decode"
								note={store.requestsMissing[model.id]
									? 'Peaks need the server’s /requests (a newer build); the line shows live values.'
									: null}
								onCopyBoost={(on) => store.setCopyBoost(on)}
								onReset={() => store.resetDecode()}
								rows={store.decodeRows}
							/>

							<RdnaDepthTile
								kind="prefill"
								onReset={() => store.resetPrefill()}
								rows={store.prefillRows}
							/>

							<RdnaCacheTile
								now={store.now}
								onReset={() => store.resetCache()}
								view={store.cacheView}
							/>

							<RdnaPriorityTile onReset={() => store.resetPriority()} prio={store.prio} />

							<RdnaRequestsTile
								history={store.history}
								missing={!!store.requestsMissing[model.id]}
								now={store.now}
								running={store.runningRows}
							/>
						{/if}
					</div>
				{/if}
			{/if}
		</main>
	</div>
{/if}

<style>
	.rf-page {
		min-height: 100vh;
		min-height: 100dvh;
		background: linear-gradient(160deg, var(--rf-bg) 0%, var(--rf-bg-deep) 100%);
		color: var(--rf-text);
		overflow-x: clip;
	}

	.rf-main {
		max-width: 1440px;
		margin: 0 auto;
		padding: 22px 32px 40px;
		box-sizing: border-box;
		display: flex;
		flex-direction: column;
		gap: 18px;
	}

	.rf-grid {
		display: flex;
		flex-wrap: wrap;
		gap: 18px;
		align-items: stretch;
	}

	/* the status tiles keep their own heights, as in the design; the classifier's stretch */
	.rf-grid-top {
		align-items: flex-start;
	}

	.rf-empty {
		margin: 40px 0;
		font-size: 15px;
		color: var(--rf-muted);
	}

	/* phones: the sidebar's floating menu button sits over the top left corner */
	@media (max-width: 767.98px) {
		.rf-main {
			padding-top: 64px;
		}
	}

	@media (max-width: 640px) {
		.rf-main {
			padding-right: 16px;
			padding-bottom: 32px;
			padding-left: 16px;
			gap: 14px;
		}

		.rf-grid {
			gap: 14px;
		}
	}
</style>
