<script lang="ts">
	/**
	 * SSD cache tile: prompt tokens served from a cache since the reset and the prefill time that
	 * saved (from finished requests), then the server's disk counters since the reset (restores,
	 * average restore, saves, evictions), the conversations on disk and the disk use. A server
	 * started without --cache-disk ({ enabled: false }) gets one centered hint instead.
	 */
	import RdnaTile from './RdnaTile.svelte';
	import type { CacheCounter } from '$lib/rdna/cache';
	import { RDNA_SSD_CACHE_DOCS_URL } from '$lib/rdna/constants';
	import { fmtBytes, fmtDuration, fmtInt, fmtSince, fmtTokens } from '$lib/rdna/format';
	import type { RdnaPromptCacheDisk } from '$lib/rdna/types';

	interface Props {
		view: {
			available: boolean;
			enabled: boolean;
			disk: RdnaPromptCacheDisk | null;
			delta: Record<CacheCounter, number> | null;
			avgRestoreMs: number | null;
			reused: number;
			savedMs: number;
			since: number;
		};
		now: number;
		onReset: () => void;
	}

	let { now, onReset, view }: Props = $props();

	let since = $derived(view.since > 0 ? fmtSince(view.since, now) : 'the server started');
	let d = $derived(view.delta);
	let usedPct = $derived(
		view.disk?.bytes && view.disk.limit_bytes
			? Math.min(100, (100 * view.disk.bytes) / view.disk.limit_bytes)
			: 0
	);
	let io = $derived(
		!d
			? '–'
			: d.bytes_read === 0 && d.bytes_written === 0
				? 'nothing read or written yet'
				: `${fmtBytes(d.bytes_read)} read · ${fmtBytes(d.bytes_written)} written`
	);
</script>

{#if view.available && !view.enabled}
	<RdnaTile fill gap={14} label="SSD prompt cache" title="SSD CACHE">
		<div class="rf-cache-off">
			<p>
				Enable <a
					class="rf-link"
					href={RDNA_SSD_CACHE_DOCS_URL}
					rel="noopener noreferrer"
					target="_blank">ssd cache</a
				> to improve agentic performance significantly
			</p>
		</div>
	</RdnaTile>
{:else}
	<RdnaTile
		gap={14}
		label="SSD prompt cache"
		{onReset}
		resetLabel="Reset SSD cache figures"
		sub="since {since}"
		title="SSD CACHE"
	>
		<div class="rf-cache-big">
			<div>
				<p class="rf-cache-reused rf-num">{fmtTokens(view.reused)}</p>

				<p class="rf-cache-cap">prompt tokens reused instead of prefilled again</p>
			</div>

			<div>
				<p class="rf-cache-saved rf-num">
					{view.savedMs > 0 ? fmtDuration(view.savedMs) : view.reused > 0 ? '–' : '0 s'}
				</p>

				<p class="rf-cache-cap">of prefill saved</p>
			</div>
		</div>

		{#if !view.available}
			<p class="rf-cache-note">
				The server does not report its SSD cache yet (a newer build adds it to /props).
			</p>
		{:else}
			<dl class="rf-cache-stats">
				<div>
					<dt>restores</dt>

					<dd>{fmtInt(d?.loads)}</dd>
				</div>

				<div>
					<dt>avg restore</dt>

					<dd>{view.avgRestoreMs !== null ? fmtDuration(view.avgRestoreMs) : '–'}</dd>
				</div>

				<div>
					<dt>saves</dt>

					<dd>{fmtInt(d?.saves)}</dd>
				</div>

				<div>
					<dt>evicted</dt>

					<dd>{fmtInt(d?.evictions)}</dd>
				</div>
			</dl>

			<div class="rf-cache-disk">
				<div class="rf-cache-line">
					<span
						>{fmtInt(view.disk?.entries)} conversations · {fmtBytes(view.disk?.bytes)} of {fmtBytes(
							view.disk?.limit_bytes
						)}</span
					>

					<span class="rf-cache-io">{io}</span>
				</div>

				<div class="rf-track"><div class="rf-fill" style="width: {usedPct}%;"></div></div>
			</div>
		{/if}
	</RdnaTile>
{/if}

<style>
	p {
		margin: 0;
	}

	.rf-cache-off {
		flex: 1;
		min-height: 140px;
		display: flex;
		align-items: center;
		justify-content: center;
		text-align: center;
		padding: 8px 12px 16px;
	}

	.rf-cache-off p {
		max-width: 360px;
		font-size: 15px;
		line-height: 1.5;
		color: var(--rf-muted);
	}

	.rf-cache-big {
		display: flex;
		flex-wrap: wrap;
		align-items: flex-end;
		gap: 10px 28px;
	}

	.rf-cache-reused {
		font-size: 46px;
		line-height: 1;
		color: var(--rf-red);
	}

	.rf-cache-saved {
		font-size: 32px;
		line-height: 1;
		color: var(--rf-text);
	}

	.rf-cache-cap {
		margin-top: 4px;
		font-size: 13px;
		color: var(--rf-text-dim);
	}

	.rf-cache-note {
		font-size: 13px;
		color: var(--rf-muted);
	}

	.rf-cache-stats {
		margin: 0;
		display: grid;
		grid-template-columns: repeat(4, minmax(0, 1fr));
		gap: 10px;
		font-size: 13px;
	}

	.rf-cache-stats dt {
		color: var(--rf-muted);
	}

	.rf-cache-stats dd {
		margin: 2px 0 0;
		font-size: 18px;
		font-weight: 800;
	}

	.rf-cache-disk {
		display: flex;
		flex-direction: column;
		gap: 5px;
	}

	.rf-cache-line {
		display: flex;
		flex-wrap: wrap;
		justify-content: space-between;
		gap: 4px 12px;
		font-size: 13px;
		color: var(--rf-text-dim);
	}

	.rf-cache-io {
		color: var(--rf-muted);
	}

	.rf-track {
		height: 8px;
		border-radius: 4px;
		background: var(--rf-track);
	}

	.rf-fill {
		height: 8px;
		border-radius: 4px;
		background: var(--rf-text-dim);
	}

	@media (max-width: 420px) {
		.rf-cache-stats {
			grid-template-columns: repeat(2, minmax(0, 1fr));
		}

		.rf-cache-reused {
			font-size: 38px;
		}
	}
</style>
