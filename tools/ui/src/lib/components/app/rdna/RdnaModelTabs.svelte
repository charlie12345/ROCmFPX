<script lang="ts">
	/**
	 * Model tabs: name, runtime tag and "classifier"; unloaded models greyed. A loading model shows a
	 * small rotating spinner in place of its avatar (no text; a static ring without motion). The tabs
	 * stay on one row; when they do not fit, the row scrolls sideways (the selected tab is kept in
	 * view) and a dropdown on the right lists every model (owner, 2026-11-03: servers with many
	 * models loaded).
	 */
	import RdnaAvatar from './RdnaAvatar.svelte';
	import { Check, ChevronDown } from '@lucide/svelte';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import type { HfView, ModelView } from '$lib/rdna/status-store.svelte';

	interface Props {
		models: ModelView[];
		selectedId: string | null;
		hf: Record<string, HfView>;
		onSelect: (id: string) => void;
	}

	let { hf, models, onSelect, selectedId }: Props = $props();

	let row = $state<HTMLElement | null>(null);
	let overflowing = $state(false);

	function measure(): void {
		if (row) overflowing = row.scrollWidth > row.clientWidth + 1;
	}

	// re-measure when the row resizes or the model list changes
	$effect(() => {
		void models.length;

		if (!row) return;

		measure();

		const ro = new ResizeObserver(measure);

		ro.observe(row);

		return () => ro.disconnect();
	});

	// keep the selected tab in view when the row scrolls
	$effect(() => {
		const id = selectedId;

		if (!row || !id) return;

		const tab = row.querySelector<HTMLElement>(`[data-model-id="${CSS.escape(id)}"]`);

		tab?.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'nearest' });
	});
</script>

<div class="rf-tabbar">
	<div bind:this={row} aria-label="Models" class="rf-tabs" role="tablist">
		{#each models as m (m.id)}
			{@const active = m.id === selectedId}
			<button
				class:rf-tab-active={active}
				class:rf-tab-off={!m.loaded && !m.loading}
				aria-selected={active}
				class="rf-tab"
				data-model-id={m.id}
				onclick={() => onSelect(m.id)}
				role="tab"
				title={m.loaded || m.loading ? m.id : `${m.id} · ${m.status}`}
				type="button"
			>
				{#if m.loading}
					<span aria-label="Loading" class="rf-tab-spinner" role="img"></span>
				{:else}
					<RdnaAvatar name={m.label} size={22} src={hf[m.id]?.avatar} />
				{/if}

				<span class="rf-tab-name">{m.label}</span>

				{#if m.runtime}
					<span class:rf-tag-on={active} class="rf-tag">{m.runtime}</span>
				{/if}

				{#if m.classifier}
					<span class:rf-tag-on={active} class="rf-tag">classifier</span>
				{/if}

				{#if !m.loaded && !m.loading}
					<span class="rf-tab-state">{m.status}</span>
				{/if}
			</button>
		{/each}
	</div>

	{#if overflowing}
		<DropdownMenu.Root>
			<DropdownMenu.Trigger>
				{#snippet child({ props })}
					<button {...props} aria-label="All models" class="rf-tabs-more" type="button">
						{models.length} models <ChevronDown class="h-4 w-4" />
					</button>
				{/snippet}
			</DropdownMenu.Trigger>

			<DropdownMenu.Content align="end" class="max-h-[60vh] min-w-64 overflow-y-auto">
				{#each models as m (m.id)}
					<DropdownMenu.Item class="gap-3" onSelect={() => onSelect(m.id)}>
						{#if m.id === selectedId}
							<Check class="h-4 w-4 shrink-0" />
						{:else}
							<span class="h-4 w-4 shrink-0"></span>
						{/if}

						<span class:rf-tab-off={!m.loaded && !m.loading} class="min-w-0 flex-1 truncate"
							>{m.label}</span
						>

						<span class="text-xs text-muted-foreground"
							>{[m.runtime, m.classifier ? 'classifier' : null, m.loaded ? null : m.status]
								.filter(Boolean)
								.join(' · ')}</span
						>
					</DropdownMenu.Item>
				{/each}
			</DropdownMenu.Content>
		</DropdownMenu.Root>
	{/if}
</div>

<style>
	.rf-tabbar {
		display: flex;
		align-items: center;
		gap: 8px;
		border-bottom: 1px solid var(--rf-border);
	}

	/* one row; scrolls sideways when the tabs do not fit (no visible scrollbar: the dropdown lists all) */
	.rf-tabs {
		display: flex;
		flex: 1 1 auto;
		min-width: 0;
		gap: 8px;
		overflow-x: auto;
		scrollbar-width: none;
	}

	.rf-tabs::-webkit-scrollbar {
		display: none;
	}

	.rf-tab {
		display: inline-flex;
		flex: none;
		align-items: center;
		gap: 10px;
		padding: 10px 16px;
		color: var(--rf-text-dim);
		background: transparent;
		border: none;
		border-bottom: 3px solid transparent;
		font-family: inherit;
		font-size: 14px;
		font-weight: 600;
		cursor: pointer;
		max-width: 360px;
	}

	.rf-tab:hover {
		color: var(--rf-text);
	}

	.rf-tab:focus-visible {
		outline: 2px solid var(--rf-orange);
		outline-offset: -2px;
	}

	.rf-tab-active {
		color: var(--rf-text);
		font-weight: 700;
		border-bottom-color: var(--rf-red);
	}

	.rf-tab-off {
		opacity: 0.5;
	}

	.rf-tab-name {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.rf-tabs-more {
		display: inline-flex;
		flex: none;
		align-items: center;
		gap: 6px;
		padding: 6px 10px;
		border-radius: 8px;
		border: 1px solid var(--rf-border);
		background: var(--rf-panel);
		color: var(--rf-text-dim);
		font-family: inherit;
		font-size: 13px;
		font-weight: 600;
		white-space: nowrap;
		cursor: pointer;
	}

	.rf-tabs-more:hover {
		color: var(--rf-text);
	}

	.rf-tabs-more:focus-visible {
		outline: 2px solid var(--rf-orange);
		outline-offset: 2px;
	}

	.rf-tag {
		padding: 2px 8px;
		border-radius: 6px;
		background: var(--rf-track);
		color: var(--rf-text-dim);
		font-size: 11px;
		font-weight: 700;
		white-space: nowrap;
	}

	.rf-tag-on {
		background: var(--rf-tag-bg);
		color: var(--rf-tag-fg);
	}

	.rf-tab-state {
		font-size: 11px;
		color: var(--rf-muted);
	}

	.rf-tab-spinner {
		flex: none;
		box-sizing: border-box;
		width: 22px;
		height: 22px;
		border-radius: 50%;
		border: 3px solid var(--rf-track);
		border-top-color: var(--rf-red);
		border-right-color: var(--rf-orange);
		animation: rfTabSpin 0.9s linear infinite;
	}

	@keyframes rfTabSpin {
		to {
			transform: rotate(360deg);
		}
	}

	/* no motion: a still ring */
	@media (prefers-reduced-motion: reduce) {
		.rf-tab-spinner {
			animation: none;
		}
	}
</style>
