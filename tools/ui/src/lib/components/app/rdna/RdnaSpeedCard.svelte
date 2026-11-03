<script lang="ts">
	/**
	 * The 16:9 speed card (ShareCard.dc.html), sized in container units so the page shows it at any
	 * width and the export draws the same card at 1200 x 675. The cheetah (card.ts): snoozing before
	 * the model has any bench value, sparking (with glowing numbers) for a fast result (>= 800 t/s
	 * prefill and >= 50 t/s decode), else walking. When a bench result changes the cheetah, the new
	 * picture reveals over the old one from the left with a feathered edge, drifting in with a brief
	 * lift (a plain crossfade without motion); the first render and a model switch show it at once.
	 * The numbers never move with it and carry no record glow (owner, 2026-11-03). The sleeping
	 * cheetah sits a little larger and further right than the walking one. The hardware label is
	 * editable and shrinks to fit its column.
	 */
	import '$lib/rdna/rdna.css';
	import RdnaCornerLogo from './RdnaCornerLogo.svelte';
	import { base } from '$app/paths';
	import { CARD_EMPTY_LINE, type Cheetah, CHEETAH_FILES, cheetahState } from '$lib/rdna/card';
	import { DASH } from '$lib/rdna/constants';
	import { cardLabelSize } from '$lib/rdna/hardware';
	import { onMount, tick } from 'svelte';

	interface Props {
		hardware: string | null;
		/**
		 * The label is settled (named, or the server answered). Until then the card's column shows a
		 * spinner with "Loading …" (no label, figures or model), so "Name your hardware" only appears
		 * when the hardware could not be named.
		 */
		hardwareKnown?: boolean;
		prefill: number | undefined;
		decode: number | undefined;
		copy: number | undefined;
		avatar: string | null;
		/** The model's NAME (model-info.ts), under the figures with its avatar. */
		modelName: string;
		footer: string;
		editable?: boolean;
		onHardwareChange?: (label: string | null) => void;
		node?: HTMLElement | null;
		/** The model the card is for: a change of model swaps the cheetah without the reveal. */
		modelKey?: string;
	}

	let {
		avatar,
		copy,
		decode,
		editable = false,
		footer,
		hardware,
		hardwareKnown = true,
		modelKey,
		modelName,
		node = $bindable(null),
		onHardwareChange,
		prefill
	}: Props = $props();

	const PLACEHOLDER = 'Name your hardware';

	/** No finished bench run yet: no numbers, one line instead. */
	let empty = $derived(prefill === undefined && decode === undefined && copy === undefined);

	let cheetah = $derived(cheetahState({ copy, decode, prefill }));
	let hot = $derived(cheetah === 'spark');

	/** The picture being revealed over (null: none), and whether the reveal runs. */
	let under = $state<Cheetah | null>(null);
	let revealing = $state(false);
	let seenCheetah: Cheetah | null = null;
	let seenModel: string | undefined;

	// a result-driven change reveals the new cheetah; the first render and a model switch do not
	$effect(() => {
		const c = cheetah;
		const m = modelKey;

		if (seenCheetah !== null && c !== seenCheetah && m === seenModel) {
			under = seenCheetah;
			revealing = false;
		} else if (m !== seenModel) {
			under = null;
			revealing = false;
		}

		seenCheetah = c;
		seenModel = m;
	});

	/** The new picture is loaded: start the reveal (never on a half-loaded picture). */
	function loaded(): void {
		if (under !== null && !revealing) revealing = true;
	}

	function revealed(): void {
		under = null;
		revealing = false;
	}

	const cheetahSrc = (c: Cheetah) => `${base}/rdna/${CHEETAH_FILES[c]}`;

	// all three pictures up front, so a reveal never shows a half-loaded one
	onMount(() => {
		for (const c of Object.keys(CHEETAH_FILES) as Cheetah[]) new Image().src = cheetahSrc(c);
	});
	let labelText = $derived(hardware?.trim() || PLACEHOLDER);
	/** The hardware is not settled yet: a spinner instead of the label, the figures and the model. */
	let loadingHw = $derived(!hardware && !hardwareKnown);
	let labelEl = $state<HTMLElement | null>(null);
	let widthAt100 = $state<number | null>(null);
	let hwSize = $derived(cardLabelSize(labelText, widthAt100));
	let editing = $state(false);
	let draft = $state('');
	let inputEl = $state<HTMLInputElement | null>(null);

	const fmt = (v: number | undefined, decimals?: number) =>
		typeof v === 'number' && Number.isFinite(v) ? v.toFixed(decimals ?? (v >= 100 ? 0 : 1)) : DASH;

	let measureCanvas: HTMLCanvasElement | null = null;

	function measure(text: string): void {
		if (!labelEl) return;

		measureCanvas ??= document.createElement('canvas');

		const ctx = measureCanvas.getContext('2d');

		if (!ctx) return;

		const cs = getComputedStyle(labelEl);

		ctx.font = `800 100px ${cs.fontFamily}`;
		widthAt100 = ctx.measureText(text).width;
	}

	$effect(() => {
		const text = labelText;

		if (!labelEl) return;

		measure(text);
		void document.fonts?.ready.then(() => measure(text));
	});

	async function startEdit(): Promise<void> {
		if (!editable) return;

		draft = hardware ?? '';
		editing = true;
		await tick();
		inputEl?.focus();
		inputEl?.select();
	}

	function finishEdit(save: boolean): void {
		if (!editing) return;

		editing = false;

		if (save) onHardwareChange?.(draft.trim() || null);
	}
</script>

<div bind:this={node} class="rf-card rf-theme-dark">
	{#if under !== null}
		<img
			class:rf-card-img-sleep={under === 'sleep'}
			class:rf-card-img-walk={under !== 'spark'}
			alt=""
			class="rf-card-img"
			data-cheetah={under}
			src={cheetahSrc(under)}
		/>
	{/if}

	<img
		class:rf-card-img-pending={under !== null && !revealing}
		class:rf-card-img-reveal={revealing}
		class:rf-card-img-sleep={cheetah === 'sleep'}
		class:rf-card-img-walk={cheetah !== 'spark'}
		alt=""
		class="rf-card-img"
		data-cheetah={cheetah}
		onanimationend={revealed}
		onerror={loaded}
		onload={loaded}
		src={cheetahSrc(cheetah)}
	/>

	<div aria-hidden="true" class="rf-card-topline"></div>

	<RdnaCornerLogo
		style="position: absolute; right: 0; top: 0.5cqw; width: 18.33cqw; height: 18.33cqw;"
	/>

	<div class="rf-card-center">
		{#if loadingHw}
			<div class="rf-card-loading" role="status">
				<span aria-hidden="true" class="rf-card-spinner"></span>Loading …
			</div>
		{:else}
			{#if editing}
				<input
					bind:this={inputEl}
					bind:value={draft}
					aria-label="Hardware label"
					class="rf-card-hw rf-card-hw-input"
					maxlength="60"
					onblur={() => finishEdit(true)}
					onkeydown={(e) => {
						if (e.key === 'Enter') finishEdit(true);
						else if (e.key === 'Escape') finishEdit(false);
					}}
					placeholder="AMD Radeon RX 7900 XTX"
					style="font-size: {hwSize}cqw;"
				/>
			{:else if editable}
				<button
					bind:this={labelEl}
					class:rf-card-hw-placeholder={!hardware}
					class="rf-card-hw rf-card-hw-btn"
					onclick={startEdit}
					style="font-size: {hwSize}cqw;"
					title="Edit the hardware label (kept in this browser)"
					type="button">{labelText}</button
				>
			{:else}
				<p bind:this={labelEl} class="rf-card-hw" style="font-size: {hwSize}cqw;">{labelText}</p>
			{/if}

			{#if empty}
				<p class="rf-card-empty">{CARD_EMPTY_LINE}</p>
			{:else}
				<p
					class:rf-card-glow-big={hot}
					class:rf-card-none={prefill === undefined}
					class="rf-card-big rf-num"
				>
					{fmt(prefill, 0)}
				</p>

				<p class="rf-card-biglabel rf-num">t/s prefill</p>

				<div aria-hidden="true" class="rf-card-divider"></div>

				<div class="rf-card-pair">
					<div>
						<p
							class:rf-card-glow-small={hot}
							class:rf-card-none={decode === undefined}
							class="rf-card-small rf-num"
						>
							{fmt(decode, 1)}<span class="rf-card-unit">&nbsp;t/s</span>
						</p>

						<p class="rf-card-smalllabel">decode</p>
					</div>

					<div>
						<p
							class:rf-card-glow-small={hot}
							class:rf-card-none={copy === undefined}
							class="rf-card-small rf-num"
						>
							{fmt(copy)}<span class="rf-card-unit">&nbsp;t/s</span>
						</p>

						<p class="rf-card-smalllabel">text copy boost</p>
					</div>
				</div>
			{/if}

			<p class="rf-card-with">
				{#if avatar}
					<img alt="" class="rf-card-avatar" src={avatar} />
				{/if}

				<strong>{modelName}</strong>
			</p>
		{/if}
	</div>

	<p class="rf-card-foot">{footer}</p>
</div>

<style>
	.rf-card {
		container-type: inline-size;
		position: relative;
		width: 100%;
		aspect-ratio: 16 / 9;
		overflow: hidden;
		border-radius: 10px;
		background: #000000;
		color: var(--rf-text);
	}

	.rf-card-img {
		position: absolute;
		left: 0;
		top: 0;
		height: 100%;
		width: auto;
		max-width: none;
	}

	.rf-card-img-walk {
		left: -7cqw;
	}

	/* the sleeping cheetah: 12% larger (anchored at the bottom, the sky above is cut) and further right */
	.rf-card-img-sleep {
		top: auto;
		bottom: 0;
		height: 112%;
		left: -3cqw;
	}

	/* the new cheetah waits, hidden, until its picture has loaded */
	.rf-card-img-pending {
		opacity: 0;
	}

	/*
	 * The reveal: left to right under a feathered edge (a quarter of the picture wide), drifting in
	 * from 2% left with a brief brightness lift, about a second, ease-out. The mask is three times
	 * the picture's width, so its edge starts left of the picture and ends right of it.
	 */
	.rf-card-img-reveal {
		-webkit-mask-image: linear-gradient(90deg, #000 45.8%, transparent 54.2%);
		mask-image: linear-gradient(90deg, #000 45.8%, transparent 54.2%);
		-webkit-mask-size: 300% 100%;
		mask-size: 300% 100%;
		-webkit-mask-repeat: no-repeat;
		mask-repeat: no-repeat;
		animation: rfCheetahReveal 1000ms cubic-bezier(0.2, 0.8, 0.2, 1) both;
	}

	@keyframes rfCheetahReveal {
		from {
			-webkit-mask-position: 100% 0;
			mask-position: 100% 0;
			transform: translateX(-2%);
			filter: brightness(1.15);
		}

		to {
			-webkit-mask-position: 0 0;
			mask-position: 0 0;
			transform: none;
			filter: brightness(1);
		}
	}

	/* no motion: a plain crossfade */
	@media (prefers-reduced-motion: reduce) {
		.rf-card-img-reveal {
			-webkit-mask-image: none;
			mask-image: none;
			animation: rfCheetahFade 300ms ease-out both;
		}
	}

	@keyframes rfCheetahFade {
		from {
			opacity: 0;
		}
	}

	.rf-card-topline {
		position: absolute;
		left: 0;
		top: 0;
		width: 100%;
		height: 0.5cqw;
		background: var(--rf-grad);
	}

	.rf-card-center {
		position: absolute;
		left: 43cqw;
		width: 39cqw;
		top: 0;
		bottom: 0;
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		text-align: center;
	}

	.rf-card-hw {
		margin: 0;
		font-weight: 800;
		letter-spacing: -0.03cqw;
		white-space: nowrap;
		line-height: 1.2;
		color: var(--rf-text);
	}

	.rf-card-hw-btn {
		background: transparent;
		border: none;
		padding: 0;
		font-family: inherit;
		cursor: text;
	}

	.rf-card-hw-btn:hover {
		text-decoration: underline dotted rgba(255, 255, 255, 0.4);
		text-underline-offset: 0.4em;
	}

	.rf-card-hw-btn:focus-visible {
		outline: 2px solid var(--rf-orange);
		outline-offset: 4px;
	}

	.rf-card-hw-placeholder {
		color: var(--rf-faint);
	}

	.rf-card-hw-input {
		width: 39cqw;
		text-align: center;
		background: rgba(0, 0, 0, 0.6);
		border: 1px solid var(--rf-border-strong);
		border-radius: 0.6cqw;
		padding: 0.2cqw 0.6cqw;
		font-family: inherit;
		outline: none;
	}

	.rf-card-big {
		margin: 0.6cqw 0 0;
		font-size: 15.5cqw;
		line-height: 0.95;
		color: var(--rf-red);
	}

	.rf-card-glow-big {
		text-shadow:
			0 0 2.4cqw rgba(237, 28, 36, 0.85),
			0 0 6cqw rgba(255, 106, 61, 0.5);
	}

	.rf-card-big.rf-card-none,
	.rf-card-small.rf-card-none {
		color: var(--rf-grey-bar);
	}

	/* the hardware is not settled yet: a brand spinner and "Loading …" in the middle of the column */
	.rf-card-loading {
		display: flex;
		align-items: center;
		gap: 1.2cqw;
		font-size: 2.2cqw;
		font-weight: 600;
		color: var(--rf-text-dim);
	}

	.rf-card-spinner {
		flex: none;
		width: 2.4cqw;
		height: 2.4cqw;
		box-sizing: border-box;
		border-radius: 50%;
		border: 0.35cqw solid rgba(255, 255, 255, 0.15);
		border-top-color: var(--rf-red);
		border-right-color: var(--rf-orange);
		animation: rfCardSpin 0.9s linear infinite;
	}

	@keyframes rfCardSpin {
		to {
			transform: rotate(360deg);
		}
	}

	@media (prefers-reduced-motion: reduce) {
		.rf-card-spinner {
			animation-duration: 2.4s;
		}
	}

	.rf-card-empty {
		margin: 4cqw 0 3cqw;
		font-size: 2.2cqw;
		font-weight: 600;
		color: var(--rf-text-dim);
	}

	.rf-card-biglabel {
		margin: 0.3cqw 0 0;
		font-size: 3.6cqw;
		color: #dadde2;
	}

	.rf-card-divider {
		width: 36cqw;
		height: 0.25cqw;
		margin: 1.8cqw 0 1.5cqw;
		background: linear-gradient(
			90deg,
			rgba(255, 255, 255, 0.75),
			#ed1c24,
			rgba(255, 106, 61, 0.15)
		);
	}

	.rf-card-pair {
		display: flex;
		gap: 5.8cqw;
		justify-content: center;
	}

	.rf-card-small {
		margin: 0;
		font-size: 5.8cqw;
		line-height: 1;
		color: var(--rf-red);
	}

	.rf-card-glow-small {
		text-shadow:
			0 0 1.4cqw rgba(237, 28, 36, 0.85),
			0 0 3.6cqw rgba(255, 106, 61, 0.45);
	}

	.rf-card-unit {
		font-family: var(--font-sans, ui-sans-serif, system-ui, sans-serif);
		font-style: normal;
		font-size: 2.2cqw;
		font-weight: 700;
		color: var(--rf-text);
		text-shadow: none;
	}

	.rf-card-smalllabel {
		margin: 0.5cqw 0 0;
		font-size: 2.15cqw;
		color: var(--rf-text-soft);
	}

	.rf-card-with {
		margin: 2.8cqw 0 0;
		display: inline-flex;
		align-items: center;
		gap: 1.1cqw;
		font-size: 2.5cqw;
		color: var(--rf-text-dim);
	}

	.rf-card-avatar {
		width: 3.6cqw;
		height: 3.6cqw;
		border-radius: 0.9cqw;
		object-fit: cover;
		background: #ffffff;
	}

	.rf-card-with strong {
		color: var(--rf-text);
		font-weight: 800;
	}

	.rf-card-foot {
		margin: 0;
		position: absolute;
		right: 2.2cqw;
		bottom: 1.4cqw;
		font-size: 1.15cqw;
		color: var(--rf-faint);
	}
</style>
