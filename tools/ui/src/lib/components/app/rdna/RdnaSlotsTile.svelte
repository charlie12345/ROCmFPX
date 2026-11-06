<script lang="ts">
	/**
	 * Slots tile: per slot a status dot (green working, amber held, grey idle; the state as tooltip),
	 * "Slot N", its priority, what it does, the live rate and its context; then the shared KV pool
	 * and the memory of the model's devices. Each slot is a title row, a detail line and the context
	 * bar: a running request's activity on the detail line, an idle slot's context size instead. A
	 * finished request stays on its slot (priority, context) until the next one starts; the dot
	 * turns grey and an idle slot shows no rate. A newer request that came and went between two
	 * polls clears it. The tile is narrow next to the speed card, so lines wrap
	 * between words only: a number stays with its unit (a no-break space). The memory line has the
	 * device name on the left and "used/total GB" on the right; when both do not fit, the figure
	 * moves whole to the next line, still on the right. While the model loads, its slots show their
	 * last state (or empty rows) and the dots pulse grey to brand red until it is ready; no text
	 * says so.
	 */
	import RdnaPill from './RdnaPill.svelte';
	import { fmtClock, fmtCtx, fmtInt, fmtRate, fmtTokens } from '$lib/rdna/format';
	import type { MemorySummary, RamSummary } from '$lib/rdna/hardware';
	import type { SlotLive, SlotPhase } from '$lib/rdna/live';
	import type { SlotMemory } from '$lib/rdna/slot-memory';
	import type { RdnaSlot } from '$lib/rdna/types';

	interface Props {
		slots: RdnaSlot[] | null;
		live: SlotLive[];
		loaded: boolean;
		/** The model is loading: pulsing dots, no note. */
		loading?: boolean;
		status: string;
		pool: { used: number; total: number } | null;
		memory: MemorySummary | null;
		/** System RAM under the VRAM line (null: not reported, or the model runs on the CPU, whose line is RAM). */
		ram?: RamSummary | null;
		hardware: string | null;
		/** Each slot's latest request, kept after it finished. */
		lastRequests?: Record<number, SlotMemory>;
	}

	let {
		hardware,
		lastRequests = {},
		live,
		loaded,
		loading = false,
		memory,
		pool,
		ram = null,
		slots,
		status
	}: Props = $props();

	interface SlotView {
		state: { color: string; text: string };
		priority: string | undefined;
		detail: string;
		rate: string;
		bar: string;
		ctx: number;
	}

	/** A running slot shows its live values; an idle one its finished request, if it had one. */
	function viewOf(s: RdnaSlot, l: SlotLive | undefined): SlotView {
		const last = lastRequests[s.id];

		if ((!l || l.phase === 'idle') && last && last.endedAt !== null) {
			return {
				bar: barOf(last.phase),
				ctx: last.ctx,
				detail: `${fmtInt(last.ctx)}\u00a0tokens in context`,
				priority: last.priority,
				rate: '–',
				state: { color: 'var(--rf-grey-bar)', text: `Idle · finished ${fmtClock(last.endedAt)}` }
			};
		}

		const ctx = l?.ctx ?? s.n_prompt_tokens ?? 0;

		if (loading && s.n_prompt_tokens === undefined) {
			// a slot of a model that is still loading, never seen before: nothing to tell yet
			return {
				bar: barOf(undefined),
				ctx: 0,
				detail: '–',
				priority: s.priority,
				rate: '–',
				state: stateOf(undefined)
			};
		}

		return {
			bar: barOf(l?.phase),
			ctx,
			detail:
				l && l.phase !== 'idle' ? detailOf(l.phase, l) : `${fmtInt(ctx)}\u00a0tokens in context`,
			priority: s.priority,
			rate: rateOf(l),
			state: stateOf(l)
		};
	}

	const GIB = 1024 * 1024 * 1024;

	function stateOf(l: SlotLive | undefined): { color: string; text: string } {
		switch (l?.phase) {
			case 'prefill':
				return { color: 'var(--rf-green)', text: 'Prefill' };
			case 'decode':
				return { color: 'var(--rf-green)', text: 'Decode' };
			case 'held':
				return { color: 'var(--rf-amber)', text: 'Held: a higher-priority request runs first' };
			default:
				return { color: 'var(--rf-grey-bar)', text: 'Idle' };
		}
	}

	function detailOf(
		phase: SlotPhase,
		l: { processed: number; cached: number; decoded: number }
	): string {
		switch (phase) {
			case 'prefill':
				return `prefill · ${fmtInt(l.processed)}\u00a0new${l.cached > 0 ? `, ${fmtInt(l.cached)}\u00a0cached` : ''}`;
			case 'decode':
				return `decode · ${fmtInt(l.decoded)}\u00a0tokens out`;
			case 'held':
				return l.decoded > 0 ? `held · ${fmtInt(l.decoded)}\u00a0tokens out` : 'held · waiting';
			default:
				return 'idle';
		}
	}

	function rateOf(l: SlotLive | undefined): string {
		if (l?.phase === 'prefill' && l.prefillTps !== null) return `${fmtRate(l.prefillTps, 0)} t/s`;

		if (l?.phase === 'decode' && l.decodeTps !== null) return `${fmtRate(l.decodeTps, 1)} t/s`;

		return '–';
	}

	function barOf(phase: SlotPhase | undefined): string {
		if (phase === 'prefill' || phase === 'decode') return 'var(--rf-grad)';

		if (phase === 'held') return 'var(--rf-grey-bar)';

		return 'var(--rf-border-strong)';
	}
</script>

<section aria-label="Slots" class="rf-slots">
	<h2>SLOTS</h2>

	{#if !loaded && !loading}
		<!-- not loaded: says so in the middle of the tile, no slots and no memory (owner, 2026-11-03) -->
		<div class="rf-off">
			<span aria-hidden="true" class="rf-off-dot"></span>

			<p class="rf-off-title">Not loaded</p>

			<p class="rf-note">This model is {status}.</p>
		</div>
	{:else if !slots}
		<p class="rf-note">–</p>
	{:else}
		{#each slots as s, i (s.id)}
			{@const v = viewOf(s, live[i])}
			{@const pct = s.n_ctx ? Math.min(100, (100 * v.ctx) / s.n_ctx) : 0}
			<div class="rf-slot">
				<div class="rf-slot-head">
					<span
						class:rf-slot-dot-loading={loading}
						aria-label={loading ? 'Starting' : v.state.text}
						class="rf-slot-dot"
						role="img"
						style={loading
							? undefined
							: `background: ${v.state.color}; box-shadow: 0 0 8px ${v.state.color};`}
						title={loading ? 'Starting' : v.state.text}
					></span>

					<strong>Slot {s.id}</strong>

					<RdnaPill priority={v.priority} />
				</div>

				<div class="rf-slot-line">
					<span>{v.detail}</span>

					<span class="rf-slot-rate">{v.rate}</span>
				</div>

				<div class="rf-track">
					<div class="rf-fill" style="width: {pct}%; background: {v.bar};"></div>
				</div>
			</div>
		{/each}

		{#if pool}
			<p class="rf-pool">
				Shared KV pool: <span class="rf-nowrap"
					><strong>{fmtTokens(pool.used)}</strong> of {fmtCtx(pool.total)}&nbsp;tokens</span
				>
			</p>
		{/if}
	{/if}

	<div class:rf-mem-off={!loaded && !loading} class="rf-mem">
		<p class="rf-mem-line">
			<span class="rf-mem-hw">{hardware ?? memory?.kind ?? 'VRAM'}</span>

			<span class="rf-mem-figure"
				>{#if memory}<strong>{(memory.used / GIB).toFixed(1)}</strong>/{(
						memory.total / GIB
					).toFixed(1)}&nbsp;GB{:else}–{/if}</span
			>
		</p>

		<div class="rf-track">
			<div
				class="rf-fill"
				style="width: {memory
					? Math.min(100, (100 * memory.used) / memory.total)
					: 0}%; background: var(--rf-text-dim);"
			></div>
		</div>

		<!-- system RAM (owner, 2026-11-07): the model's own share bright, the rest of the system dim -->
		{#if ram && memory?.kind !== 'RAM'}
			<p
				class="rf-mem-line rf-mem-ram"
				title={ram.own !== null
					? `This model: ${(ram.own / GIB).toFixed(1)} GB, everything else: ${((ram.used - ram.own) / GIB).toFixed(1)} GB`
					: undefined}
			>
				<span class="rf-mem-hw">RAM</span>

				<span class="rf-mem-figure"
					><strong>{(ram.used / GIB).toFixed(1)}</strong>/{(ram.total / GIB).toFixed(
						1
					)}&nbsp;GB</span
				>
			</p>

			<div class="rf-track rf-ram-track">
				<div
					class="rf-fill rf-ram-rest"
					style="width: {Math.min(100, (100 * ram.used) / ram.total)}%;"
				></div>

				{#if ram.own !== null}
					<div
						class="rf-fill rf-ram-own"
						style="width: {Math.min(
							100,
							(100 * ram.own) / ram.total
						)}%; background: var(--rf-text-dim);"
					></div>
				{/if}
			</div>
		{/if}
	</div>
</section>

<style>
	.rf-slots {
		flex: 1 1 300px;
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

	h2 {
		margin: 0;
		font-size: 14px;
		font-weight: 800;
		letter-spacing: 2.4px;
		color: var(--rf-label);
	}

	.rf-off {
		flex: 1;
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		gap: 8px;
		min-height: 180px;
		text-align: center;
	}

	.rf-off-dot {
		width: 12px;
		height: 12px;
		border-radius: 50%;
		background: var(--rf-grey-bar);
	}

	.rf-off-title {
		margin: 0;
		font-size: 18px;
		font-weight: 700;
		color: var(--rf-text-dim);
	}

	/* a model that is not loaded has no memory to show */
	.rf-mem-off {
		display: none;
	}

	.rf-note {
		margin: 0;
		font-size: 13px;
		color: var(--rf-muted);
	}

	.rf-slot {
		display: flex;
		flex-direction: column;
		gap: 6px;
		padding-bottom: 10px;
		border-bottom: 1px solid var(--rf-track);
	}

	.rf-slot-head {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 8px;
		font-size: 15px;
	}

	.rf-slot-dot {
		flex: none;
		width: 10px;
		height: 10px;
		border-radius: 50%;
		cursor: help;
	}

	/* a loading model: the dots pulse slowly from grey to the brand red */
	.rf-slot-dot-loading {
		animation: rfSlotStarting 2.4s ease-in-out infinite;
	}

	@keyframes rfSlotStarting {
		0%,
		100% {
			background: var(--rf-grey-bar);
			box-shadow: 0 0 0 transparent;
		}

		50% {
			background: var(--rf-red);
			box-shadow: 0 0 10px var(--rf-glow-bar-hot);
		}
	}

	/* no motion: a still brand-red dot */
	@media (prefers-reduced-motion: reduce) {
		.rf-slot-dot-loading {
			animation: none;
			background: var(--rf-red);
			box-shadow: 0 0 8px var(--rf-glow-bar-hot);
		}
	}

	.rf-slot-line {
		display: flex;
		justify-content: space-between;
		gap: 10px;
		font-size: 13px;
		color: var(--rf-text-dim);
	}

	.rf-pool strong,
	.rf-mem-line strong {
		color: var(--rf-text);
	}

	.rf-slot-rate {
		font-weight: 700;
		color: var(--rf-text);
		white-space: nowrap;
	}

	.rf-track {
		height: 8px;
		border-radius: 4px;
		background: var(--rf-track);
	}

	.rf-fill {
		height: 8px;
		border-radius: 4px;
		transition: width 0.8s ease;
	}

	.rf-pool {
		margin: 14px 0 0;
		font-size: 13px;
		color: var(--rf-text-dim);
	}

	.rf-mem {
		display: flex;
		flex-direction: column;
		gap: 6px;
		margin-top: auto;
	}

	/* device name left, figure right; a figure that does not fit moves to the next line, right */
	.rf-mem-line {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		column-gap: 10px;
		margin: 0;
		font-size: 13px;
		color: var(--rf-text-dim);
	}

	.rf-mem-hw {
		flex: 1 1 auto;
		min-width: 0;
		color: var(--rf-muted);
	}

	.rf-mem-figure {
		flex: none;
		margin-left: auto;
		white-space: nowrap;
	}

	.rf-mem-ram {
		margin-top: 4px;
	}

	/* the model's share over the whole system's use */
	.rf-ram-track {
		position: relative;
	}

	.rf-ram-rest {
		background: var(--rf-text-dim);
		opacity: 0.35;
	}

	.rf-ram-own {
		position: absolute;
		left: 0;
		top: 0;
		bottom: 0;
	}

	/* a value with its unit never breaks inside */
	.rf-nowrap {
		white-space: nowrap;
	}

	@media (max-width: 640px) {
		.rf-slots {
			padding: 16px;
		}
	}
</style>
