<script lang="ts">
	/**
	 * RDNA-FAST wordmark, one line: silver "RDNA-" with a grey outline (graphite with a dark outline
	 * in the light theme), red "FAST" with a dark red outline, heavy italic Exo 2. The tagline sits
	 * under it as in the design. The colours are tokens (rdna.css).
	 */
	import '$lib/rdna/rdna.css';
	import { RDNA_NAME, RDNA_TAGLINE } from '$lib/rdna/constants';
	import { ensureRdnaFonts } from '$lib/rdna/fonts';
	import { onMount } from 'svelte';

	interface Props {
		width?: number;
		tagline?: boolean;
		class?: string;
	}

	let { class: className = '', tagline = true, width = 176 }: Props = $props();

	const uid = $props.id();
	let height = $derived(Math.round((width * 36) / 176));

	onMount(ensureRdnaFonts);
</script>

<span class="rf-wordmark {className}">
	<svg aria-label={RDNA_NAME} {height} role="img" viewBox="0 0 196 40" {width}>
		<defs>
			<linearGradient id="{uid}-silver" x1="0" x2="0" y1="0" y2="1">
				<stop offset="0" style="stop-color: var(--rf-wm-silver-0);" />

				<stop offset="0.45" style="stop-color: var(--rf-wm-silver-1);" />

				<stop offset="0.55" style="stop-color: var(--rf-wm-silver-2);" />

				<stop offset="1" style="stop-color: var(--rf-wm-silver-3);" />
			</linearGradient>

			<linearGradient id="{uid}-red" x1="0" x2="0" y1="0" y2="1">
				<stop offset="0" style="stop-color: var(--rf-wm-red-0);" />

				<stop offset="0.55" style="stop-color: var(--rf-wm-red-1);" />

				<stop offset="1" style="stop-color: var(--rf-wm-red-2);" />
			</linearGradient>
		</defs>

		<text
			font-family="'Exo 2', sans-serif"
			font-size="32"
			font-style="italic"
			font-weight="900"
			letter-spacing="0.4"
			paint-order="stroke"
			stroke-linejoin="round"
			stroke-width="2"
			x="3"
			y="31"
			><tspan fill="url(#{uid}-silver)" style="stroke: var(--rf-wm-silver-stroke);">RDNA-</tspan
			><tspan fill="url(#{uid}-red)" style="stroke: var(--rf-wm-red-stroke);">FAST</tspan></text
		>
	</svg>

	{#if tagline}
		<span class="rf-tagline">{RDNA_TAGLINE}</span>
	{/if}
</span>

<style>
	.rf-wordmark {
		display: inline-flex;
		flex-direction: column;
		line-height: 1;
	}

	.rf-tagline {
		font-size: 11px;
		letter-spacing: 2.5px;
		color: var(--rf-muted);
		margin: 2px 0 0 3px;
	}
</style>
