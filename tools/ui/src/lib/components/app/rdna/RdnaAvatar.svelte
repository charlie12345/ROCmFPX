<script lang="ts">
	/** A model's Hugging Face avatar (a data URL), or its first letter on the brand gradient. */
	interface Props {
		src: string | null | undefined;
		name: string;
		size?: number;
		radius?: number;
		alt?: string;
	}

	let { alt = '', name, radius, size = 22, src }: Props = $props();

	let r = $derived(radius ?? Math.round(size / 4));
	let letter = $derived((name.trim()[0] ?? '?').toUpperCase());
</script>

{#if src}
	<img
		{alt}
		class="rf-avatar"
		height={size}
		{src}
		style="width: {size}px; height: {size}px; border-radius: {r}px;"
		width={size}
	/>
{:else}
	<span
		aria-hidden={alt ? undefined : 'true'}
		aria-label={alt || undefined}
		class="rf-avatar rf-avatar-letter"
		role={alt ? 'img' : undefined}
		style="width: {size}px; height: {size}px; border-radius: {r}px; font-size: {Math.round(
			size * 0.5
		)}px;">{letter}</span
	>
{/if}

<style>
	.rf-avatar {
		flex: none;
		object-fit: cover;
		background: var(--rf-avatar-bg);
	}

	.rf-avatar-letter {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		background: var(--rf-grad);
		color: var(--rf-on-grad);
		font-weight: 800;
	}
</style>
