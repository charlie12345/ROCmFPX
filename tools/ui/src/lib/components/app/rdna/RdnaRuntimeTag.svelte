<script lang="ts">
	/** Runtime tag (ROCm / Vulkan / CUDA / CPU) of a model in the chat's model picker, from the router's args. */
	import '$lib/rdna/rdna.css';
	import { modelArgs, runtimeTag } from '$lib/rdna/model-info';
	import type { RdnaModelEntry } from '$lib/rdna/types';
	import { modelsStore } from '$lib/stores';

	interface Props {
		model: string | null | undefined;
	}

	let { model }: Props = $props();

	let runtime = $derived.by(() => {
		if (!model) return null;

		const entry = modelsStore.routerModels.find(
			(m) => m.id === model || m.aliases?.includes(model)
		);

		return entry ? runtimeTag(modelArgs(entry as unknown as RdnaModelEntry)) : null;
	});
</script>

{#if runtime}
	<span class="rf-runtime-tag" title="Runtime">{runtime}</span>
{/if}

<style>
	.rf-runtime-tag {
		flex: none;
		padding: 1px 7px;
		border-radius: 6px;
		background: var(--rf-tag-bg);
		color: var(--rf-tag-fg);
		font-size: 10.5px;
		font-weight: 700;
		line-height: 1.5;
		white-space: nowrap;
	}
</style>
