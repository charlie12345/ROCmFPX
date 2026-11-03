<script lang="ts">
	/**
	 * Priority pill: xhigh = the red-orange brand gradient with white text, high = red outline,
	 * default = grey fill, low = grey outline.
	 */
	interface Props {
		priority: string | null | undefined;
	}

	let { priority }: Props = $props();

	let p = $derived(priority || 'default');
	let kind = $derived(['xhigh', 'high', 'default', 'low'].includes(p) ? p : 'default');
</script>

<span class="rf-pill rf-pill-{kind}">{p}</span>

<style>
	.rf-pill {
		display: inline-block;
		justify-self: start;
		padding: 2px 9px;
		border-radius: 999px;
		font-size: 12px;
		font-weight: 700;
		line-height: 1.35;
		border: 1px solid transparent;
		white-space: nowrap;
	}

	.rf-pill-xhigh {
		background: var(--rf-grad);
		color: var(--rf-on-grad);
		border-color: transparent;
	}

	.rf-pill-high {
		background: transparent;
		color: var(--rf-label);
		border-color: var(--rf-red);
	}

	.rf-pill-default {
		background: var(--rf-border);
		color: var(--rf-text-dim);
		border-color: var(--rf-border);
	}

	.rf-pill-low {
		background: transparent;
		color: var(--rf-muted);
		border-color: var(--rf-border-strong);
	}
</style>
