<script lang="ts">
	/**
	 * Decode or Prefill by context depth: the translucent white line with the bold italic number on
	 * top is the live value (from /slots polls); the red bar is the peak. When the line goes past the
	 * bar, the bar jumps to it and pulses for about five seconds (a new record); a finished request
	 * the polls never caught raises it by its exact rate. Reset clears the peaks.
	 */
	import RdnaTile from './RdnaTile.svelte';
	import { niceScale } from '$lib/rdna/depth';
	import { fmtRate } from '$lib/rdna/format';
	import type { DepthRow } from '$lib/rdna/status-store.svelte';

	interface Props {
		kind: 'decode' | 'prefill';
		rows: DepthRow[];
		onReset: () => void;
		note?: string | null;
		/** Decode: the "Text copy boost" toggle (peaks with fast text copies or without). */
		copyBoost?: boolean;
		onCopyBoost?: (on: boolean) => void;
	}

	let { copyBoost = false, kind, note = null, onCopyBoost, onReset, rows }: Props = $props();

	let decimals = $derived(kind === 'decode' ? 1 : 0);
	let scale = $derived(niceScale(Math.max(0, ...rows.map((r) => r.peak ?? 0))));
	let fallbackScale = $derived(kind === 'decode' ? 100 : 1000);
	let s = $derived(
		rows.some((r) => r.peak !== null)
			? scale
			: Math.max(fallbackScale, ...rows.map((r) => r.live ?? 0))
	);

	function pct(v: number | null): number {
		return v === null ? 0 : Math.min(100, (100 * v) / s);
	}
</script>

<RdnaTile
	actions={kind === 'decode' && onCopyBoost ? copyToggle : undefined}
	label={kind === 'decode' ? 'Decode' : 'Prefill'}
	{onReset}
	resetLabel={kind === 'decode' ? 'Reset decode peaks' : 'Reset prefill peaks'}
	sub="peak t/s"
	title={kind === 'decode' ? 'DECODE' : 'PREFILL'}
>
	{#if note}
		<p class="rf-depth-note">{note}</p>
	{/if}

	<div class="rf-depth-rows">
		{#each rows as row (row.hi)}
			{@const shown = row.live}
			{@const on = shown !== null}
			<div class="rf-depth-row">
				<span class="rf-depth-label">{row.label}</span>

				<div class="rf-depth-track">
					<div
						class:rf-record-bar={row.record}
						class="rf-depth-peak"
						style="width: {pct(row.peak)}%;"
					></div>

					<div
						aria-hidden="true"
						class="rf-depth-line"
						style="left: {pct(shown)}%; opacity: {on ? 1 : 0};"
					></div>

					<span class="rf-depth-live rf-num" style="left: {pct(shown)}%; opacity: {on ? 1 : 0};"
						>{on ? fmtRate(shown, decimals) : ''}</span
					>
				</div>

				<div class="rf-depth-value">{fmtRate(row.peak, decimals)}</div>
			</div>
		{/each}
	</div>
</RdnaTile>

{#snippet copyToggle()}
	<button
		class:rf-copy-on={copyBoost}
		aria-pressed={copyBoost}
		class="rf-reset rf-copy"
		onclick={() => onCopyBoost?.(!copyBoost)}
		title={copyBoost
			? 'Fast text copies (nearly every drafted token accepted) count in the decode peaks'
			: 'Fast text copies are left out of the decode peaks'}
		type="button"
	>
		<svg
			aria-hidden="true"
			fill={copyBoost ? 'currentColor' : 'none'}
			height="13"
			stroke="currentColor"
			stroke-linejoin="round"
			stroke-width="2"
			viewBox="0 0 24 24"
			width="13"><path d="M13 2 4 14h7l-1 8 9-12h-7z" /></svg
		>Text copy boost
	</button>
{/snippet}

<style>
	.rf-copy-on {
		color: var(--rf-orange);
	}

	.rf-depth-note {
		margin: 0;
		font-size: 13px;
		color: var(--rf-muted);
	}

	/*
	 * Rows sit close: 6px between them plus the 24px above each bar that holds the live number
	 * (18px high, 6px over the bar), so the number never reaches the row above.
	 */
	.rf-depth-rows {
		display: flex;
		flex-direction: column;
		gap: 6px;
	}

	.rf-depth-row {
		display: grid;
		grid-template-columns: 96px minmax(0, 1fr) 76px;
		align-items: end;
		gap: 14px;
	}

	.rf-depth-label {
		font-size: 14px;
		line-height: 18px;
		color: var(--rf-text-soft);
	}

	.rf-depth-track {
		position: relative;
		height: 18px;
		margin-top: 24px;
		border-radius: 5px;
		background: var(--rf-track);
	}

	.rf-depth-peak {
		position: absolute;
		left: 0;
		top: 0;
		bottom: 0;
		border-radius: 5px;
		background: var(--rf-grad);
		box-shadow: 0 0 14px var(--rf-bar-shadow);
		transition: width 0.8s ease;
	}

	.rf-depth-line {
		position: absolute;
		top: -4px;
		bottom: -4px;
		width: 2px;
		margin-left: -1px;
		border-radius: 1px;
		background: var(--rf-live-line);
		box-shadow: 0 0 6px var(--rf-live-line-glow);
		transition:
			left 0.8s ease,
			opacity 0.3s ease;
	}

	.rf-depth-live {
		position: absolute;
		bottom: calc(100% + 6px);
		transform: translateX(-50%);
		font-size: 18px;
		line-height: 1;
		color: var(--rf-live-num);
		text-shadow: 0 0 10px var(--rf-live-num-glow);
		white-space: nowrap;
		transition:
			left 0.8s ease,
			opacity 0.3s ease;
	}

	.rf-depth-value {
		text-align: right;
		font-size: 16px;
		line-height: 18px;
		font-weight: 800;
	}

	@media (max-width: 640px) {
		.rf-depth-row {
			grid-template-columns: 74px minmax(0, 1fr) 52px;
			gap: 10px;
		}

		.rf-depth-label {
			font-size: 13px;
		}
	}
</style>
