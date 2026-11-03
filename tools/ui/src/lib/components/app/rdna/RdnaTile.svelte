<script lang="ts">
	/**
	 * A stat tile of the status page: red spaced title, muted subtitle, content, and a subtle Reset
	 * at the bottom right when the tile can be reset.
	 */
	import '$lib/rdna/rdna.css';
	import type { Snippet } from 'svelte';

	interface Props {
		label: string;
		title: string;
		sub?: string;
		flex?: string;
		gap?: number;
		/** Stretch to the row's height (for a body centered in the tile). */
		fill?: boolean;
		resetLabel?: string;
		onReset?: () => void;
		/** Small controls left of Reset, in the same subtle style. */
		actions?: Snippet;
		children: Snippet;
	}

	let {
		actions,
		children,
		fill = false,
		flex = '1 1 520px',
		gap = 16,
		label,
		onReset,
		resetLabel,
		sub,
		title
	}: Props = $props();
</script>

<section
	aria-label={label}
	class="rf-tile"
	style="flex: {flex}; gap: {gap}px;{fill ? ' align-self: stretch;' : ''}"
>
	<div class="rf-tile-head">
		<h2>{title}</h2>

		{#if sub}
			<span class="rf-tile-sub">{sub}</span>
		{/if}
	</div>

	{@render children()}

	{#if onReset || actions}
		<div class="rf-tile-foot">
			{@render actions?.()}

			{#if onReset}
				<button
					aria-label={resetLabel ?? `Reset ${label}`}
					class="rf-reset"
					onclick={onReset}
					type="button"
				>
					<svg
						aria-hidden="true"
						fill="none"
						height="13"
						stroke="currentColor"
						stroke-linecap="round"
						stroke-linejoin="round"
						stroke-width="2"
						viewBox="0 0 24 24"
						width="13"><path d="M3 12a9 9 0 1 0 3-6.7" /><path d="M3 4v5h5" /></svg
					>Reset
				</button>
			{/if}
		</div>
	{/if}
</section>

<style>
	.rf-tile {
		min-width: 0;
		border-radius: 16px;
		border: 1px solid var(--rf-border);
		background: var(--rf-panel);
		padding: 20px 22px;
		display: flex;
		flex-direction: column;
		box-sizing: border-box;
	}

	.rf-tile-head {
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

	.rf-tile-sub {
		font-size: 13px;
		color: var(--rf-muted);
	}

	.rf-tile-foot {
		display: flex;
		flex-wrap: wrap;
		justify-content: flex-end;
		align-items: center;
		gap: 4px;
		margin-top: auto;
	}

	@media (max-width: 640px) {
		.rf-tile {
			padding: 16px;
		}
	}
</style>
