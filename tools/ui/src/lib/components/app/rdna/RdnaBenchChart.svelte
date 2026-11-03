<script lang="ts">
	/**
	 * The bench chart: this run against the one before, group by group (Main.dc.html). A run in
	 * progress fills in as its requests finish; a figure it improves glows. The title is the model's
	 * own NAME (as in the model header and the tabs) on the hardware; the subtitle the quant and the
	 * bench details, like the header's NAME / quant.
	 */
	import RdnaAvatar from './RdnaAvatar.svelte';
	import RdnaCornerLogo from './RdnaCornerLogo.svelte';
	import { chartGroups } from '$lib/rdna/bench/metrics';
	import type { BenchRun } from '$lib/rdna/bench/types';
	import { fmtCtx, fmtDay } from '$lib/rdna/format';

	interface Props {
		now: BenchRun | null;
		prev: BenchRun | null;
		hardware: string | null;
		avatar: string | null;
		/** The model's NAME: the GGUF file stem without its quant (model-info.ts). */
		modelName: string;
		/** Its quant (IQ4_XS), if it has one. */
		quant: string | null;
		node?: HTMLElement | null;
		/** Does this run's figure glow (it just improved)? */
		glowing?: (key: string) => boolean;
	}

	let {
		avatar,
		glowing = () => false,
		hardware,
		modelName,
		node = $bindable(null),
		now,
		prev,
		quant
	}: Props = $props();

	let subtitle = $derived.by(() => {
		const bench = `bench${now?.nCtx ? ` at ${fmtCtx(now.nCtx)} context` : ''}, this run against the one before`;

		return quant ? `${quant} · ${bench}` : `B${bench.slice(1)}`;
	});

	let groups = $derived(chartGroups(now, prev));

	function runNote(run: BenchRun): string {
		if (run.inProgress) return ' · in progress';

		return `${run.incomplete ? ' · incomplete' : ''}${run.overlapped ? ' · shared the server' : ''}`;
	}

	function fmtValue(v: number | null, unit: string): string {
		if (v === null) return '–';

		if (unit === 's') return `${v.toFixed(1)} s`;

		return `${v >= 100 ? Math.round(v) : v.toFixed(1)} ${unit}`;
	}

	function scaleOf(rows: { now: number | null; prev: number | null }[]): number {
		const all = rows.flatMap((r) => [r.now ?? 0, r.prev ?? 0]);

		return Math.max(1, ...all);
	}
</script>

<div bind:this={node} class="rf-chart rf-theme-dark">
	<div aria-hidden="true" class="rf-chart-topline"></div>

	<RdnaCornerLogo
		shadow={false}
		style="position: absolute; right: 0; top: 4px; width: 150px; height: 150px;"
	/>

	<div class="rf-chart-head">
		<RdnaAvatar name={modelName} radius={11} size={44} src={avatar} />

		<div>
			<p class="rf-chart-title">{modelName}{hardware ? ` on ${hardware}` : ''}</p>

			<p class="rf-chart-sub">{subtitle}</p>
		</div>
	</div>

	{#if !now}
		<p class="rf-chart-empty">
			No bench run yet. Run bench measures this model with two scripted agent sessions, three code
			edits and eighteen answers, at priority xhigh (other requests are held while it runs).
		</p>
	{:else}
		<div class="rf-chart-legend">
			<span class="rf-legend-now"
				><span class="rf-swatch rf-swatch-now"></span>this run · {fmtDay(now.t)}{runNote(now)}</span
			>

			{#if prev}
				<span class="rf-legend-prev"
					><span class="rf-swatch rf-swatch-prev"></span>previous run · {fmtDay(prev.t)}{runNote(
						prev
					)}</span
				>
			{/if}
		</div>

		{#each groups as g (g.title)}
			{@const scale = scaleOf(g.rows)}
			<div class="rf-group">
				<div class="rf-group-head">
					<span class="rf-group-title">{g.title}</span>

					<span class="rf-group-sub">{g.sub}</span>
				</div>

				{#each g.rows as r (r.label)}
					<div class="rf-row">
						<span class="rf-row-label">{r.label}</span>

						<div class="rf-row-bars">
							<div class="rf-bar-line">
								<div
									class:rf-record-bar={glowing(r.key)}
									class="rf-bar rf-bar-now"
									style="width: {r.now === null ? 0 : (88 * r.now) / scale}%;"
								></div>

								<span class="rf-bar-value">{fmtValue(r.now, g.unit)}</span>
							</div>

							{#if prev}
								<div class="rf-bar-line">
									<div
										class="rf-bar rf-bar-prev"
										style="width: {r.prev === null ? 0 : (88 * r.prev) / scale}%;"
									></div>

									<span class="rf-bar-value rf-bar-value-prev">{fmtValue(r.prev, g.unit)}</span>
								</div>
							{/if}
						</div>
					</div>
				{/each}
			</div>
		{/each}

		{#if now.skipped.length > 0}
			<p class="rf-chart-note">Left out (the context is too small): {now.skipped.join(', ')}.</p>
		{/if}
	{/if}
</div>

<style>
	.rf-chart {
		position: relative;
		overflow: hidden;
		border-radius: 10px;
		background: linear-gradient(160deg, #0b0b0d, #17070a);
		padding: 26px 28px 20px;
		display: flex;
		flex-direction: column;
		gap: 14px;
		color: var(--rf-text);
	}

	/* the brand line along the top, as on the speed card */
	.rf-chart-topline {
		position: absolute;
		left: 0;
		top: 0;
		width: 100%;
		height: 4px;
		background: var(--rf-grad);
	}

	.rf-chart p {
		margin: 0;
	}

	.rf-chart-head {
		display: flex;
		align-items: center;
		gap: 14px;
		padding-right: 150px;
	}

	.rf-chart-title {
		font-size: 26px;
		font-weight: 800;
		letter-spacing: -0.4px;
	}

	.rf-chart-sub {
		margin-top: 3px !important;
		font-size: 14px;
		color: var(--rf-muted);
	}

	.rf-chart-empty {
		font-size: 14px;
		line-height: 1.6;
		color: var(--rf-text-dim);
		max-width: 640px;
		padding: 24px 0 40px;
	}

	.rf-chart-legend {
		display: flex;
		flex-wrap: wrap;
		gap: 8px 18px;
		font-size: 13px;
	}

	.rf-legend-now {
		display: inline-flex;
		align-items: center;
		gap: 8px;
		font-weight: 600;
	}

	.rf-legend-prev {
		display: inline-flex;
		align-items: center;
		gap: 8px;
		color: var(--rf-muted);
	}

	.rf-swatch {
		width: 22px;
		height: 11px;
		border-radius: 3px;
	}

	.rf-swatch-now {
		background: var(--rf-grad);
	}

	.rf-swatch-prev {
		background: var(--rf-grey-bar);
	}

	.rf-group {
		border-radius: 12px;
		border: 1px solid var(--rf-border);
		background: var(--rf-panel);
		padding: 12px 16px;
		display: flex;
		flex-direction: column;
		gap: 10px;
	}

	.rf-group-head {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: 4px 12px;
	}

	.rf-group-title {
		font-size: 13px;
		font-weight: 800;
		letter-spacing: 2px;
		color: var(--rf-label);
	}

	.rf-group-sub {
		font-size: 12px;
		color: var(--rf-muted);
	}

	.rf-row {
		display: grid;
		grid-template-columns: 210px minmax(0, 1fr);
		align-items: center;
		gap: 14px;
	}

	.rf-row-label {
		font-size: 14px;
		color: var(--rf-text-soft);
	}

	.rf-row-bars {
		display: flex;
		flex-direction: column;
		gap: 4px;
		min-width: 0;
	}

	.rf-bar-line {
		display: flex;
		align-items: center;
		gap: 10px;
	}

	.rf-bar {
		position: relative;
		height: 13px;
		border-radius: 4px;
		flex: none;
		transition: width 0.8s ease;
	}

	.rf-bar-now {
		background: var(--rf-grad);
		box-shadow: 0 0 10px var(--rf-bar-shadow);
	}

	.rf-bar-prev {
		background: var(--rf-grey-bar);
	}

	.rf-bar-value {
		font-size: 14px;
		font-weight: 800;
		white-space: nowrap;
	}

	.rf-bar-value-prev {
		font-size: 13px;
		font-weight: 400;
		color: var(--rf-muted);
	}

	.rf-chart-note {
		font-size: 12px;
		color: var(--rf-faint);
	}

	@media (max-width: 640px) {
		.rf-chart {
			padding: 18px 16px 16px;
		}

		.rf-chart-head {
			padding-right: 90px;
		}

		.rf-chart-title {
			font-size: 19px;
		}

		.rf-row {
			grid-template-columns: minmax(0, 1fr);
			gap: 4px;
		}

		.rf-chart :global(.rf-corner) {
			width: 90px !important;
			height: 90px !important;
		}
	}
</style>
