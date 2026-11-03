<script lang="ts">
	/**
	 * Model header: avatar, NAME (file stem without quant) and "<quant> · <HF repo>" (the repo linked
	 * only when the Hub confirms it), then chips: parameters, size, context and slots, modalities.
	 */
	import RdnaAvatar from './RdnaAvatar.svelte';
	import { HF_BASE_URL } from '$lib/constants/huggingface.constants';
	import { fmtBytes, fmtCtx, fmtParams } from '$lib/rdna/format';
	import type { HfView, ModelView } from '$lib/rdna/status-store.svelte';
	import type { RdnaProps } from '$lib/rdna/types';

	interface Props {
		model: ModelView;
		hf: HfView | null;
		props: RdnaProps | null;
		nCtx: number | null;
		totalSlots: number | null;
	}

	let { hf, model, nCtx, props, totalSlots }: Props = $props();

	let repo = $derived(model.naming?.repo ?? null);
	let quant = $derived(model.naming?.quant ?? null);
	let repoExists = $derived(!!hf?.repo?.exists && !!repo);
	let meta = $derived(model.entry.meta ?? null);
	let chips = $derived.by(() => {
		const out: string[] = [];

		if (meta?.n_params) out.push(`${fmtParams(meta.n_params)} parameters`);

		if (meta?.size) out.push(fmtBytes(meta.size));

		if (nCtx) {
			const slots = totalSlots ? ` · ${totalSlots} ${totalSlots === 1 ? 'slot' : 'slots'}` : '';

			out.push(`${fmtCtx(nCtx)} context${slots}`);
		}

		if (props?.modalities?.vision) out.push('vision');

		if (props?.modalities?.audio) out.push('audio');

		return out;
	});
</script>

<section aria-label="Model" class="rf-model">
	<div class="rf-model-id">
		<RdnaAvatar alt={model.label} name={model.label} radius={14} size={56} src={hf?.avatar} />

		<div class="rf-model-text">
			<h1>{model.naming?.name ?? model.id}</h1>

			<p>
				{#if quant}
					<span>{quant}</span>
				{/if}

				{#if quant && repoExists}
					<span aria-hidden="true">·</span>
				{/if}

				{#if repoExists && repo}
					<a
						class="rf-link rf-model-repo"
						href="{HF_BASE_URL}/{repo}"
						rel="noopener noreferrer"
						target="_blank">{repo}</a
					>
				{/if}

				{#if !quant && !repoExists}
					<span>{model.id}</span>
				{/if}

				{#if !model.loaded && !model.loading}
					<span class="rf-model-state">· {model.status}</span>
				{/if}
			</p>
		</div>
	</div>

	{#if chips.length > 0}
		<div class="rf-chips">
			{#each chips as chip (chip)}
				<span class="rf-chip">{chip}</span>
			{/each}
		</div>
	{/if}
</section>

<style>
	.rf-model {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		justify-content: space-between;
		gap: 16px;
	}

	.rf-model-id {
		display: flex;
		align-items: center;
		gap: 16px;
		min-width: 0;
	}

	.rf-model-text {
		min-width: 0;
	}

	h1 {
		margin: 0;
		font-size: 30px;
		font-weight: 800;
		letter-spacing: -0.5px;
		overflow-wrap: anywhere;
	}

	p {
		margin: 4px 0 0;
		color: var(--rf-muted);
		font-size: 15px;
		overflow-wrap: anywhere;
	}

	.rf-model-state {
		color: var(--rf-amber);
	}

	/* the HF repo link: grey like the line it sits in, with a subtle underline (owner, 2026-11-03) */
	.rf-model-repo {
		color: inherit;
		text-decoration: underline;
		text-decoration-thickness: 1px;
		text-underline-offset: 3px;
		text-decoration-color: color-mix(in srgb, currentColor 35%, transparent);
	}

	.rf-model-repo:hover {
		color: var(--rf-text-dim);
		text-decoration-color: currentColor;
	}

	.rf-chips {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
	}

	.rf-chip {
		padding: 6px 11px;
		border-radius: 8px;
		background: var(--rf-panel);
		border: 1px solid var(--rf-border);
		font-size: 13px;
		color: var(--rf-text-dim);
	}

	@media (max-width: 640px) {
		h1 {
			font-size: 22px;
		}
	}
</style>
