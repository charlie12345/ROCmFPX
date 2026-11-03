<script lang="ts">
	/**
	 * Requests tile (completion models): what runs now (from /slots) on top, then the last finished
	 * requests this browser saw (from /requests), newest first, 50 rows; the page grows with it and
	 * the table scrolls sideways in its box on a phone.
	 */
	import RdnaPill from './RdnaPill.svelte';
	import { REQUESTS_SHOWN } from '$lib/rdna/constants';
	import { fmtClock, fmtDuration, fmtInt, fmtRate, fmtTokens } from '$lib/rdna/format';
	import { decodeRate, isFastCopy, prefillRate } from '$lib/rdna/history';
	import type { RunningRow } from '$lib/rdna/status-store.svelte';
	import type { CompletionRecord, RequestRecord } from '$lib/rdna/types';

	interface Props {
		history: RequestRecord[];
		running: RunningRow[];
		missing: boolean;
		now: number;
	}

	let { history, missing, now, running }: Props = $props();

	interface Row {
		key: string;
		time: string;
		priority: string;
		ctx: string;
		newCached: string;
		prefill: string;
		out: string;
		decode: string;
		runtime: string;
		status: string;
		statusColor: string;
	}

	const STATUS_COLORS: Record<string, string> = {
		cancelled: 'var(--rf-amber)',
		decode: 'var(--rf-green)',
		done: 'var(--rf-muted)',
		error: 'var(--rf-label)',
		held: 'var(--rf-amber)',
		prefill: 'var(--rf-green)'
	};

	function finishedRow(r: CompletionRecord): Row {
		const fast = isFastCopy(r);
		const status = r.status ?? 'done';

		return {
			ctx: fmtTokens(r.n_ctx || (r.n_prompt ?? 0) + (r.n_gen ?? 0)),
			decode: fmtRate(decodeRate(r), 1),
			key: `${r.id}:${r.t_arrival_ms}`,
			newCached:
				r.n_processed !== undefined || r.n_cache !== undefined
					? `${fmtInt(r.n_processed ?? 0)} · ${fmtInt(r.n_cache ?? 0)}`
					: '–',
			out: r.n_gen !== undefined ? fmtInt(r.n_gen) : '–',
			prefill: (r.n_processed ?? 0) > 0 ? fmtRate(prefillRate(r), 0) : '–',
			priority: r.priority ?? 'default',
			runtime: fmtDuration(r.t_end_ms - r.t_arrival_ms),
			status: fast && status === 'done' ? 'done · fast copy' : status,
			statusColor: STATUS_COLORS[status] ?? 'var(--rf-muted)',
			time: fmtClock(r.t_start_ms ?? r.t_arrival_ms)
		};
	}

	let rows = $derived.by(() => {
		const live: Row[] = running.map((r) => ({
			ctx: fmtTokens(r.ctx),
			decode: r.phase === 'decode' ? fmtRate(r.decodeTps, 1) : '–',
			key: `run:${r.key}`,
			newCached: `${fmtInt(r.processed)} · ${fmtInt(r.cached)}`,
			out: r.decoded > 0 ? fmtInt(r.decoded) : '–',
			prefill: r.phase === 'prefill' ? fmtRate(r.prefillTps, 0) : '–',
			priority: r.priority,
			runtime: fmtDuration(now - r.t),
			status: `slot ${r.slot} · ${r.phase}`,
			statusColor: STATUS_COLORS[r.phase] ?? 'var(--rf-green)',
			time: fmtClock(r.t)
		}));
		const done = history
			.filter((r): r is CompletionRecord => r.kind === 'completion')
			.slice(0, Math.max(0, REQUESTS_SHOWN - live.length))
			.map(finishedRow);

		return [...live, ...done];
	});
</script>

<section aria-label="Requests" class="rf-req">
	<div class="rf-req-head">
		<h2>REQUESTS</h2>

		<span>the last {REQUESTS_SHOWN} this browser saw, newest first</span>
	</div>

	{#if missing && rows.length === 0}
		<p class="rf-req-note">The server keeps no request list yet (a newer build adds /requests).</p>
	{:else if rows.length === 0}
		<p class="rf-req-note">No requests yet.</p>
	{:else}
		<div aria-label="Requests table" class="rf-req-scroll" role="region" tabindex="-1">
			<table>
				<!-- fixed tracks: rows never shift as statuses and numbers change; the last fills -->
				<colgroup>
					<col style="width: 84px;" />

					<col style="width: 88px;" />

					<col style="width: 84px;" />

					<col style="width: 140px;" />

					<col style="width: 84px;" />

					<col style="width: 76px;" />

					<col style="width: 84px;" />

					<col style="width: 84px;" />

					<col style="width: 168px;" />

					<col />
				</colgroup>

				<thead>
					<tr>
						<th scope="col">STARTED</th>

						<th scope="col">PRIORITY</th>

						<th class="rf-r" scope="col">CONTEXT</th>

						<th class="rf-r" scope="col">NEW · CACHED</th>

						<th class="rf-r" scope="col">PREFILL</th>

						<th class="rf-r" scope="col">OUT</th>

						<th class="rf-r" scope="col">DECODE</th>

						<th class="rf-r" scope="col">TIME</th>

						<th scope="col">STATUS</th>

						<th aria-hidden="true"></th>
					</tr>
				</thead>

				<tbody>
					{#each rows as r (r.key)}
						<tr>
							<td class="rf-dim">{r.time}</td>

							<td><RdnaPill priority={r.priority} /></td>

							<td class="rf-r rf-soft">{r.ctx}</td>

							<td class="rf-r rf-soft" title={r.newCached}>{r.newCached}</td>

							<td class="rf-r rf-bold">{r.prefill}</td>

							<td class="rf-r rf-soft">{r.out}</td>

							<td class="rf-r rf-bold">{r.decode}</td>

							<td class="rf-r rf-soft">{r.runtime}</td>

							<td class="rf-status" style="color: {r.statusColor};" title={r.status}>{r.status}</td>

							<td aria-hidden="true"></td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{/if}
</section>

<style>
	.rf-req {
		flex: 1 1 100%;
		min-width: 0;
		border-radius: 16px;
		border: 1px solid var(--rf-border);
		background: var(--rf-panel);
		padding: 20px 22px;
		display: flex;
		flex-direction: column;
		gap: 12px;
		box-sizing: border-box;
	}

	.rf-req-head {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: 4px 12px;
	}

	h2 {
		margin: 0;
		font-size: 14px;
		font-weight: 800;
		letter-spacing: 2.4px;
		color: var(--rf-label);
	}

	.rf-req-head span,
	.rf-req-note {
		font-size: 13px;
		color: var(--rf-muted);
	}

	.rf-req-note {
		margin: 0;
	}

	.rf-req-scroll {
		overflow-x: auto;
	}

	table {
		width: 100%;
		min-width: 892px;
		table-layout: fixed;
		border-collapse: collapse;
		font-variant-numeric: tabular-nums;
	}

	th {
		padding: 8px 10px;
		font-size: 12px;
		font-weight: 700;
		letter-spacing: 0.6px;
		color: var(--rf-muted);
		border-bottom: 1px solid var(--rf-border);
		text-align: left;
		white-space: nowrap;
	}

	td {
		padding: 9px 10px;
		font-size: 14px;
		border-bottom: 1px solid var(--rf-track);
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}

	th {
		overflow: hidden;
		text-overflow: ellipsis;
	}

	.rf-r {
		text-align: right;
	}

	.rf-dim {
		color: var(--rf-muted);
	}

	.rf-soft {
		color: var(--rf-text-dim);
	}

	.rf-bold {
		font-weight: 700;
	}

	.rf-status {
		font-size: 13px;
		font-weight: 700;
	}

	@media (max-width: 640px) {
		.rf-req {
			padding: 16px;
		}
	}
</style>
