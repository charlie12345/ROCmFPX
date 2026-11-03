<script lang="ts">
	/**
	 * Chat sidebar: a live server-status card (busy slots, decode t/s now against the model's decode
	 * peak from the status page) linking to the status page, and "RDNA-FAST on GitHub". It polls
	 * /slots of the chat's model once a second while it is shown, and only when that model is loaded.
	 */
	import '$lib/rdna/rdna.css';
	import { modelQuery, rdnaGet } from '$lib/rdna/api';
	import { POLL_SLOTS_MS, RDNA_GITHUB_URL, RDNA_ROUTE_STATUS } from '$lib/rdna/constants';
	import { fmtRate } from '$lib/rdna/format';
	import { LiveRates, type SlotLive } from '$lib/rdna/live';
	import { loadCopyBoost, loadPeaks } from '$lib/rdna/storage';
	import type { RdnaSlot } from '$lib/rdna/types';
	import { modelsStore, serverStore } from '$lib/stores';
	import { onDestroy, onMount } from 'svelte';

	let live = $state.raw<SlotLive[]>([]);
	let reachable = $state(true);
	let timer: ReturnType<typeof setTimeout> | null = null;
	let alive = false;
	const rates = new LiveRates();
	let lastModel: string | null = null;

	let routerMode = $derived(serverStore.isRouterMode);
	let modelId = $derived.by(() => {
		if (!routerMode) return modelsStore.activeModelId;

		const active = modelsStore.activeModelId;

		if (active && modelsStore.getModelStatus(active) === 'loaded') return active;

		return modelsStore.routerModels.find((m) => m.status?.value === 'loaded')?.id ?? null;
	});
	let peak = $derived.by(() => {
		if (!modelId) return null;

		const peaks = loadPeaks(modelId);
		// the set the Decode tile shows: with fast text copies only when its toggle is on
		const set = loadCopyBoost(modelId) ? peaks.decode : peaks.decodeNoCopy;
		const p = Object.values(set).map((x) => x.v);

		return p.length > 0 ? Math.max(...p) : null;
	});
	let busy = $derived(live.filter((l) => l.phase !== 'idle').length);
	let decodeNow = $derived.by(() => {
		const v = live.map((l) => l.decodeTps).filter((x): x is number => x !== null);

		return v.length > 0 ? v.reduce((a, b) => a + b, 0) : null;
	});
	let pct = $derived(decodeNow !== null && peak ? Math.min(100, (100 * decodeNow) / peak) : 0);

	async function poll(): Promise<void> {
		timer = null;

		if (!alive) return;

		const id = modelId;

		if (id !== lastModel) {
			rates.reset();
			live = [];
			lastModel = id;
		}

		if (id && !document.hidden) {
			const res = await rdnaGet<RdnaSlot[]>('/slots', modelQuery(id, routerMode));

			reachable = res.status !== 0;

			if (res.ok && Array.isArray(res.data) && id === modelId) {
				live = rates.update(res.data, Date.now());
			}
		}

		if (alive) timer = setTimeout(poll, POLL_SLOTS_MS);
	}

	onMount(() => {
		alive = true;
		void poll();
	});

	onDestroy(() => {
		alive = false;

		if (timer) clearTimeout(timer);
	});
</script>

<div class="rf-side-status">
	<a class="rf-side-card" href={RDNA_ROUTE_STATUS}>
		<span class="rf-side-row">
			<span class="rf-side-title">Server status</span>

			<span class="rf-side-busy">
				<span
					aria-hidden="true"
					class="rf-side-dot"
					style="background: {!reachable
						? 'var(--rf-red)'
						: busy > 0
							? 'var(--rf-green)'
							: 'var(--rf-grey-bar)'};"
				></span>
				{!reachable
					? 'offline'
					: busy > 0
						? `${busy} ${busy === 1 ? 'slot' : 'slots'} busy`
						: 'idle'}
			</span>
		</span>

		<span class:rf-side-none={decodeNow === null} class="rf-side-num rf-num"
			>{decodeNow !== null ? fmtRate(decodeNow, 1) : '–'}<span class="rf-side-unit"
				>&nbsp;t/s now{peak !== null ? ` · peak ${fmtRate(peak, 1)}` : ''}</span
			></span
		>

		<span aria-hidden="true" class="rf-side-track"
			><span class="rf-side-fill" style="width: {pct}%;"></span></span
		>
	</a>

	<a class="rf-side-gh" href={RDNA_GITHUB_URL} rel="noopener noreferrer" target="_blank">
		<svg
			aria-hidden="true"
			fill="none"
			height="16"
			stroke="currentColor"
			stroke-linecap="round"
			stroke-linejoin="round"
			stroke-width="2"
			viewBox="0 0 24 24"
			width="16"
			><path
				d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.4 5.4 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4"
			/><path d="M9 18c-4.51 2-5-2-7-2" /></svg
		>RDNA-FAST on GitHub
	</a>
</div>

<style>
	.rf-side-status {
		display: flex;
		flex-direction: column;
		gap: 6px;
		padding: 8px 0 4px;
	}

	.rf-side-card {
		display: flex;
		flex-direction: column;
		gap: 8px;
		padding: 12px;
		border-radius: 12px;
		border: 1px solid var(--rf-warm-border);
		background: var(--rf-warm-card);
		text-decoration: none;
		color: var(--rf-text);
	}

	.rf-side-card:hover {
		border-color: var(--rf-warm-border-strong);
	}

	.rf-side-card:focus-visible,
	.rf-side-gh:focus-visible {
		outline: 2px solid var(--rf-orange);
		outline-offset: 2px;
	}

	.rf-side-row {
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: 8px;
	}

	.rf-side-title {
		font-size: 13px;
		font-weight: 700;
	}

	.rf-side-busy {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		font-size: 12px;
		color: var(--rf-muted);
	}

	.rf-side-dot {
		width: 7px;
		height: 7px;
		border-radius: 50%;
	}

	.rf-side-num {
		font-size: 26px;
		color: var(--rf-red);
		line-height: 1;
	}

	.rf-side-none {
		color: var(--rf-grey-bar);
	}

	.rf-side-unit {
		font-family: var(--font-sans, ui-sans-serif, system-ui, sans-serif);
		font-style: normal;
		font-size: 13px;
		font-weight: 600;
		color: var(--rf-text-dim);
	}

	.rf-side-track {
		display: block;
		height: 4px;
		border-radius: 2px;
		background: var(--rf-track);
	}

	.rf-side-fill {
		display: block;
		height: 4px;
		border-radius: 2px;
		background: var(--rf-grad);
		transition: width 0.8s ease;
	}

	.rf-side-gh {
		display: inline-flex;
		align-items: center;
		gap: 10px;
		padding: 9px 10px;
		border-radius: 8px;
		color: var(--rf-text-dim);
		text-decoration: none;
		font-size: 14px;
	}

	.rf-side-gh:hover {
		color: var(--rf-text);
		background: var(--rf-track);
	}
</style>
