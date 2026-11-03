<script lang="ts">
	import { ensureRdnaFonts } from '$lib/rdna/fonts';
	import { serverStore } from '$lib/stores';

	ensureRdnaFonts(); // [RDNA-FAST] the greeting in the brand's display font

	interface Props {
		isEmpty: boolean;
	}

	let { isEmpty = false }: Props = $props();
</script>

<div
	class={[
		'pointer-events-none mb-4 hidden px-4 text-center text-balance',
		isEmpty && 'mb-[calc(50dvh-8rem)] md:mb-8 pointer-events-auto block!'
	]}
>
	<h1 class="rf-greeting mb-2 text-2xl font-semibold tracking-tight md:text-3xl">Hello there</h1>

	<p class="text-muted-foreground md:text-lg">
		{serverStore.props?.modalities?.audio ? 'Record audio, type a message ' : 'Type a message'} or upload
		files to get started
	</p>
</div>
