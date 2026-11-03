<script lang="ts">
	/**
	 * The red corner triangle with the two-line wordmark (silver "RDNA-", indented red "FAST"
	 * hanging out of the triangle a little) over a soft dark shadow, as make_chart.py draws it.
	 */
	interface Props {
		style?: string;
		shadow?: boolean;
	}

	let { shadow = true, style = '' }: Props = $props();

	const uid = $props.id();
</script>

<svg aria-hidden="true" class="rf-corner" {style} viewBox="0 0 220 220">
	<defs>
		<linearGradient id="{uid}-corner" x1="0" x2="1" y1="0" y2="1">
			<stop offset="0" stop-color="#ED1C24" />

			<stop offset="1" stop-color="#FF6A3D" />
		</linearGradient>

		<linearGradient id="{uid}-silver" x1="0" x2="0" y1="0" y2="1">
			<stop offset="0" stop-color="#FFFFFF" />

			<stop offset="0.45" stop-color="#E4E6EA" />

			<stop offset="0.55" stop-color="#A7ACB4" />

			<stop offset="1" stop-color="#DADDE2" />
		</linearGradient>

		<linearGradient id="{uid}-name" x1="0" x2="0" y1="0" y2="1">
			<stop offset="0" stop-color="#FF4A3D" />

			<stop offset="0.55" stop-color="#ED1C24" />

			<stop offset="1" stop-color="#B8121A" />
		</linearGradient>

		<filter height="170%" id="{uid}-shadow" width="140%" x="-20%" y="-30%">
			<feGaussianBlur in="SourceAlpha" stdDeviation="7" />

			<feOffset dy="3" result="b" />

			<feFlood flood-color="#000000" flood-opacity="0.35" />

			<feComposite in2="b" operator="in" result="g" />

			<feMerge>
				<feMergeNode in="g" />

				<feMergeNode in="SourceGraphic" />
			</feMerge>
		</filter>
	</defs>

	<path d="M0 0 H220 V220 Z" fill="url(#{uid}-corner)" opacity="0.95" />

	<text
		fill="url(#{uid}-silver)"
		filter={shadow ? `url(#${uid}-shadow)` : undefined}
		font-family="'Exo 2', sans-serif"
		font-size="46"
		font-style="italic"
		font-weight="900"
		letter-spacing="0.5"
		paint-order="stroke"
		stroke="#6B7079"
		stroke-linejoin="round"
		stroke-width="3"
		x="27"
		y="74">RDNA-</text
	>

	<text
		fill="url(#{uid}-name)"
		filter={shadow ? `url(#${uid}-shadow)` : undefined}
		font-family="'Exo 2', sans-serif"
		font-size="46"
		font-style="italic"
		font-weight="900"
		letter-spacing="0.5"
		paint-order="stroke"
		stroke="#7A0A10"
		stroke-linejoin="round"
		stroke-width="3"
		x="74"
		y="124">FAST</text
	>
</svg>

<style>
	.rf-corner {
		overflow: visible;
	}
</style>
