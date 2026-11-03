<script lang="ts">
	/** The RDNA-FAST wordmark at the top of the expanded chat sidebar. */
	import RdnaWordmark from './RdnaWordmark.svelte';
</script>

<a aria-label="RDNA-FAST, new chat" class="rf-side-brand" href="#/">
	<RdnaWordmark width={150} />
</a>

<style>
	.rf-side-brand {
		display: inline-flex;
		margin: 0 auto 0 8px;
		padding: 2px 4px;
		border-radius: 8px;
		text-decoration: none;
		min-width: 0;
	}

	.rf-side-brand:focus-visible {
		outline: 2px solid var(--rf-orange);
		outline-offset: 2px;
	}
</style>
