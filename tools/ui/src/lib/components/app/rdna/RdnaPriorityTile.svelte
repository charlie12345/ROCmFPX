<script lang="ts">
	/** Priority tile: requests per priority and their average time to the first token, waiting included. */
	import RdnaPill from './RdnaPill.svelte';
	import RdnaTile from './RdnaTile.svelte';
	import { PRIORITIES } from '$lib/rdna/constants';
	import { niceScale } from '$lib/rdna/depth';
	import { fmtInt } from '$lib/rdna/format';
	import type { PriorityMemory } from '$lib/rdna/types';

	interface Props {
		prio: PriorityMemory;
		onReset: () => void;
	}

	let { onReset, prio }: Props = $props();

	const BARS: Record<string, string> = {
		default: 'var(--rf-text-dim)',
		high: 'var(--rf-red)',
		low: 'var(--rf-grey-bar)',
		xhigh: 'var(--rf-grad)'
	};

	let rows = $derived(
		PRIORITIES.map((p) => {
			const s = prio.byPriority[p];
			const ttft = s && s.ttftN > 0 ? s.ttftSum / s.ttftN : null;

			return { n: s?.n ?? 0, name: p, ttft };
		})
	);
	let scale = $derived(niceScale(Math.max(0, ...rows.map((r) => r.ttft ?? 0)) / 1000));
</script>

<RdnaTile
	gap={14}
	label="Request priority"
	{onReset}
	resetLabel="Reset priority figures"
	sub="requests and average time to first token, waiting included"
	title="PRIORITY"
>
	<div class="rf-prio-grid rf-prio-head">
		<span></span>

		<span class="rf-right">REQUESTS</span>

		<span>TIME TO FIRST TOKEN</span>

		<span></span>
	</div>

	{#each rows as r (r.name)}
		<div class="rf-prio-grid">
			<RdnaPill priority={r.name} />

			<span class="rf-right rf-prio-count">{fmtInt(r.n)}</span>

			<div class="rf-prio-track">
				<div
					class="rf-prio-fill"
					style="width: {r.ttft === null
						? 0
						: Math.min(100, (100 * r.ttft) / 1000 / scale)}%; background: {BARS[r.name]};"
				></div>
			</div>

			<span class="rf-right rf-prio-ttft"
				>{r.ttft === null ? '–' : `${(r.ttft / 1000).toFixed(1)} s`}</span
			>
		</div>
	{/each}
</RdnaTile>

<style>
	.rf-prio-grid {
		display: grid;
		grid-template-columns: 80px 70px minmax(0, 1fr) 64px;
		gap: 14px;
		align-items: center;
	}

	.rf-prio-head {
		gap: 10px 14px;
		font-size: 12px;
		font-weight: 700;
		letter-spacing: 0.6px;
		color: var(--rf-muted);
	}

	.rf-right {
		text-align: right;
	}

	.rf-prio-count {
		font-size: 16px;
		font-weight: 800;
		font-variant-numeric: tabular-nums;
	}

	.rf-prio-ttft {
		font-size: 15px;
		font-weight: 800;
		font-variant-numeric: tabular-nums;
	}

	.rf-prio-track {
		height: 14px;
		border-radius: 4px;
		background: var(--rf-track);
	}

	.rf-prio-fill {
		height: 14px;
		border-radius: 4px;
	}

	@media (max-width: 420px) {
		.rf-prio-grid {
			grid-template-columns: 64px 58px minmax(0, 1fr) 52px;
			gap: 8px;
		}

		.rf-prio-head {
			gap: 8px;
			font-size: 10px;
			letter-spacing: 0;
		}
	}
</style>
