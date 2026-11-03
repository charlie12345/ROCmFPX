<script lang="ts">
	/**
	 * Classifier tab: Speed (median and p95 answer time, answers per minute, peak prompt t/s), Question
	 * types (one row per type and option count) and the request list (never the question text).
	 */
	import RdnaPill from './RdnaPill.svelte';
	import RdnaTile from './RdnaTile.svelte';
	import { REQUESTS_SHOWN } from '$lib/rdna/constants';
	import { fmtClock, fmtLatency, fmtRate } from '$lib/rdna/format';
	import {
		answersPerMinute,
		median,
		percentile,
		questionTypeKey,
		questionTypeLabel,
		questionTypeRows
	} from '$lib/rdna/history';
	import type { ClassifierMemory, ClassifierRecord, RequestRecord } from '$lib/rdna/types';

	interface Props {
		classifier: ClassifierMemory;
		history: RequestRecord[];
		missing: boolean;
		now: number;
		onResetSpeed: () => void;
		onResetTypes: () => void;
		/** The speed figures that set a new best just now (record-glow.svelte.ts). */
		glowFastest?: boolean;
		glowPrompt?: boolean;
	}

	let {
		classifier,
		glowFastest = false,
		glowPrompt = false,
		history,
		missing,
		now,
		onResetSpeed,
		onResetTypes
	}: Props = $props();

	const sec = (ms: number | null) => (ms === null ? '–' : (ms / 1000).toFixed(ms < 10 ? 3 : 2));

	let p50 = $derived(median(classifier.latencies));
	let p95 = $derived(percentile(classifier.latencies, 95));
	let perMin = $derived(answersPerMinute(classifier, now));
	let types = $derived(questionTypeRows(classifier.types));
	let maxType = $derived(Math.max(1, ...types.map((t) => t.count)));
	let rows = $derived(
		history.filter((r): r is ClassifierRecord => r.kind === 'classifier').slice(0, REQUESTS_SHOWN)
	);

	function typeLabel(r: ClassifierRecord): string {
		const key = questionTypeKey(r);
		const label = key ? questionTypeLabel(key) : '–';
		const n = r.questions?.length ?? r.n_questions ?? 1;

		return n > 1 ? `${label} +${n - 1}` : label;
	}

	function probLabel(r: ClassifierRecord): string {
		if (r.type === 'score') return 'score';

		return typeof r.prob === 'number' ? r.prob.toFixed(2) : '–';
	}
</script>

<RdnaTile
	flex="3 1 560px"
	label="Speed"
	onReset={onResetSpeed}
	resetLabel="Reset classifier speed figures"
	sub="time per answer, answers per minute and peak prefill speed"
	title="SPEED"
>
	<div class="rf-cls-speed">
		<div>
			<p class="rf-cls-big rf-num rf-red">{sec(p50)}<span class="rf-unit">&nbsp;s</span></p>

			<p class="rf-cls-cap">median answer</p>
		</div>

		<div>
			<p class="rf-cls-big rf-num">{sec(p95)}<span class="rf-unit">&nbsp;s</span></p>

			<p class="rf-cls-cap">95% of answers within</p>
		</div>

		<div>
			<p class:rf-record={glowFastest} class="rf-cls-big rf-num">
				{sec(classifier.fastest)}<span class="rf-unit">&nbsp;s</span>
			</p>

			<p class="rf-cls-cap">fastest answer</p>
		</div>

		<div>
			<p class="rf-cls-big rf-num">{perMin === null ? '–' : perMin.toFixed(1)}</p>

			<p class="rf-cls-cap">answers per minute</p>
		</div>

		<div>
			<p class:rf-record={glowPrompt} class="rf-cls-big rf-num">
				{fmtRate(classifier.promptPeak, 0)}<span class="rf-unit">&nbsp;t/s</span>
			</p>

			<p class="rf-cls-cap">peak prefill</p>
		</div>
	</div>
</RdnaTile>

<RdnaTile
	flex="2 1 380px"
	gap={14}
	label="Question types"
	onReset={onResetTypes}
	resetLabel="Reset question type counts"
	sub="answers since reset"
	title="QUESTION TYPES"
>
	{#if types.length === 0}
		<p class="rf-cls-note">No answers since the reset.</p>
	{/if}

	{#each types as t (t.key)}
		<div class="rf-cls-type">
			<span class="rf-cls-type-label">{t.label}</span>

			<div class="rf-cls-track">
				<div class="rf-cls-fill" style="width: {(100 * t.count) / maxType}%;"></div>
			</div>

			<span class="rf-cls-type-count">{t.count}</span>
		</div>
	{/each}
</RdnaTile>

<section aria-label="Requests" class="rf-cls-req">
	<div class="rf-cls-head">
		<h2>REQUESTS</h2>

		<span>the last {REQUESTS_SHOWN}, newest first · questions are not shown</span>
	</div>

	{#if rows.length === 0}
		<p class="rf-cls-note">
			{missing
				? 'The server keeps no request list yet (a newer build adds /requests).'
				: 'No answers yet.'}
		</p>
	{:else}
		<div aria-label="Classifier requests table" class="rf-cls-scroll" role="region" tabindex="-1">
			<table>
				<!-- fixed tracks: rows never shift as answers and times change; the last fills -->
				<colgroup>
					<col style="width: 84px;" />

					<col style="width: 88px;" />

					<col style="width: 150px;" />

					<col style="width: 280px;" />

					<col style="width: 90px;" />

					<col style="width: 84px;" />

					<col />
				</colgroup>

				<thead>
					<tr>
						<th scope="col">STARTED</th>

						<th scope="col">PRIORITY</th>

						<th scope="col">TYPE</th>

						<th scope="col">ANSWER</th>

						<th class="rf-r" scope="col">TIME</th>

						<th scope="col">SOURCE</th>

						<th aria-hidden="true"></th>
					</tr>
				</thead>

				<tbody>
					{#each rows as r (`${r.id}:${r.t_arrival_ms}`)}
						<tr>
							<td class="rf-dim">{fmtClock(r.t_arrival_ms)}</td>

							<td><RdnaPill priority={r.priority} /></td>

							<td class="rf-soft" title={typeLabel(r)}>{typeLabel(r)}</td>

							<td title={r.status === 'error' ? 'error' : `${r.answer || '–'} · ${probLabel(r)}`}>
								{#if r.status === 'error'}
									<strong class="rf-err">error</strong>
								{:else}
									<strong>{r.answer || '–'}</strong>

									<span class="rf-dim">· {probLabel(r)}</span>
								{/if}
							</td>

							<td class="rf-r rf-bold">{r.status === 'error' ? '–' : fmtLatency(r.latency_ms)}</td>

							<td class="rf-dim">{r.status === 'error' ? '–' : r.cached ? 'cache' : 'model'}</td>

							<td aria-hidden="true"></td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{/if}
</section>

<style>
	p {
		margin: 0;
	}

	.rf-cls-speed {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(132px, 1fr));
		gap: 18px;
	}

	.rf-cls-big {
		font-size: 40px;
		line-height: 1;
		color: var(--rf-text);
		white-space: nowrap;
	}

	.rf-red {
		color: var(--rf-red);
	}

	.rf-unit {
		font-family: var(--font-sans, ui-sans-serif, system-ui, sans-serif);
		font-style: normal;
		font-size: 15px;
		font-weight: 700;
		color: var(--rf-text);
	}

	.rf-cls-cap {
		margin-top: 6px;
		font-size: 13px;
		color: var(--rf-text-dim);
	}

	.rf-cls-note {
		font-size: 13px;
		color: var(--rf-muted);
	}

	.rf-cls-type {
		display: grid;
		grid-template-columns: 110px minmax(0, 1fr) 56px;
		gap: 14px;
		align-items: center;
	}

	.rf-cls-type-label {
		font-size: 14px;
		color: var(--rf-text-soft);
	}

	.rf-cls-track {
		height: 14px;
		border-radius: 4px;
		background: var(--rf-track);
	}

	.rf-cls-fill {
		height: 14px;
		border-radius: 4px;
		background: var(--rf-grad);
	}

	.rf-cls-type-count {
		text-align: right;
		font-size: 16px;
		font-weight: 800;
		font-variant-numeric: tabular-nums;
	}

	.rf-cls-req {
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

	.rf-cls-head {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: 4px 12px;
	}

	.rf-cls-head span {
		font-size: 13px;
		color: var(--rf-muted);
	}

	h2 {
		margin: 0;
		font-size: 14px;
		font-weight: 800;
		letter-spacing: 2.4px;
		color: var(--rf-label);
	}

	.rf-cls-scroll {
		overflow-x: auto;
	}

	table {
		width: 100%;
		min-width: 776px;
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

	.rf-err {
		color: var(--rf-label);
	}

	@media (max-width: 640px) {
		.rf-cls-speed {
			grid-template-columns: repeat(2, minmax(0, 1fr));
		}

		.rf-cls-big {
			font-size: 34px;
		}

		.rf-cls-req {
			padding: 16px;
		}
	}
</style>
