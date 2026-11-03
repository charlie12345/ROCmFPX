/**
 * RDNA-FAST status page: what a model is, from the router's /v1/models entry (the child's command line
 * or its preset) and /props: runtime tag, classifier tag, name, quant and the Hugging Face repo guess.
 *
 * Name rules (owner, 2026-11-03): NAME = the GGUF file stem without its quant suffix; quant = the
 * model_ftype before " - "; repo = <folder above the model folder>/<NAME>-GGUF.
 */

import type { RdnaDevice, RdnaModelEntry } from './types';

/** The value of a command-line option (`--device ROCm0` or `--device=ROCm0`). */
export function argValue(
	args: readonly string[] | undefined,
	...names: string[]
): string | undefined {
	if (!args) return undefined;

	for (let i = 0; i < args.length; i++) {
		const a = args[i];

		for (const n of names) {
			if (a === n) return args[i + 1];

			if (a.startsWith(`${n}=`)) return a.slice(n.length + 1);
		}
	}

	return undefined;
}

export function hasArg(args: readonly string[] | undefined, ...names: string[]): boolean {
	if (!args) return false;

	return args.some((a) => names.some((n) => a === n || a.startsWith(`${n}=`)));
}

/**
 * The router's preset text (`key = value` lines) as command-line style arguments, so an unloaded
 * model (no `args`) still tells its device, model file and classifier options.
 */
export function presetArgs(preset: string | undefined): string[] {
	if (!preset) return [];

	const out: string[] = [];

	for (const line of preset.split(/\r?\n/)) {
		const m = /^\s*([A-Za-z0-9][A-Za-z0-9-]*)\s*=\s*(.*?)\s*$/.exec(line);

		if (!m) continue;

		out.push(`--${m[1]}`, m[2]);
	}

	return out;
}

/** How many slots a model's command line asks for (`-np` / `--parallel`), if it says. */
export function slotCountOf(args: readonly string[] | undefined): number | null {
	const n = Number(argValue(args, '-np', '--parallel'));

	return Number.isInteger(n) && n >= 1 && n <= 256 ? n : null;
}

/** The arguments of a model entry: the running child's command line, else its preset. */
export function modelArgs(entry: RdnaModelEntry | undefined): string[] {
	if (!entry?.status) return [];

	if (entry.status.args && entry.status.args.length > 0) return entry.status.args;

	return presetArgs(entry.status.preset);
}

const RUNTIME_NAMES: Record<string, string> = {
	cann: 'CANN',
	cpu: 'CPU',
	cuda: 'CUDA',
	hip: 'ROCm',
	metal: 'Metal',
	mtl: 'Metal',
	musa: 'MUSA',
	none: 'CPU',
	opencl: 'OpenCL',
	rocm: 'ROCm',
	rpc: 'RPC',
	sycl: 'SYCL',
	vulkan: 'Vulkan',
	webgpu: 'WebGPU'
};

/** Runtime name of one backend device name: `ROCm0` -> ROCm, `Vulkan1` -> Vulkan, `none` -> CPU. */
export function runtimeOfDevice(device: string): string {
	const d = device.trim();
	const m = /^([A-Za-z]+)/.exec(d);
	const key = (m ? m[1] : d).toLowerCase();

	return RUNTIME_NAMES[key] ?? (m ? m[1] : d);
}

/**
 * The runtime tag of a model from its `--device` list (`ROCm0,ROCm1` -> ROCm, `none` -> CPU);
 * `--n-gpu-layers 0` without a device list means CPU. Null when the arguments do not say.
 */
export function runtimeTag(args: readonly string[] | undefined): string | null {
	const dev = argValue(args, '--device', '-dev');

	if (dev !== undefined && dev.trim() !== '') {
		const names = Array.from(
			new Set(
				dev
					.split(',')
					.map((d) => d.trim())
					.filter(Boolean)
					.map(runtimeOfDevice)
			)
		);

		return names.length > 0 ? names.join(' + ') : null;
	}

	const ngl = argValue(args, '--n-gpu-layers', '-ngl', '--gpu-layers');

	if (ngl !== undefined && Number(ngl) === 0) return 'CPU';

	return null;
}

/** The runtime tag from /props devices, for servers whose command line is unknown. */
export function runtimeFromDevices(devices: readonly RdnaDevice[] | undefined): string | null {
	if (!devices || devices.length === 0) return null;

	const names = Array.from(new Set(devices.map((d) => runtimeOfDevice(d.name ?? d.type ?? 'CPU'))));

	return names.join(' + ');
}

/** A classifier model carries --classifier-head or --classifier-config. */
export function isClassifier(args: readonly string[] | undefined): boolean {
	return hasArg(args, '--classifier-head', '--classifier-config');
}

/** The model file of a child: `-m` / `--model`. */
export function modelPathOf(args: readonly string[] | undefined): string | undefined {
	return argValue(args, '-m', '--model');
}

/** `IQ4_XS - 4.25 bpw` -> `IQ4_XS`; `BF16` -> `BF16`. */
export function quantFromFtype(ftype: string | undefined | null): string | null {
	if (!ftype) return null;

	const q = ftype.split(' - ')[0].trim();

	return q || null;
}

/** A quant tag at the end of a file stem, after `-`, `_` or `.`. */
const QUANT_SUFFIX_REGEX =
	/[-_.]((?:UD-)?(?:I?Q\d+(?:_[A-Z0-9]+)*|TQ\d+_\d+|BF16|F16|F32|FP16|FP32|FP8|MXFP\d+(?:_[A-Z0-9]+)*|ROCmFP\d+(?:-FP\d+)*))$/i;
const SHARD_REGEX = /-\d{5}-of-\d{5}$/i;

export interface ParsedModelPath {
	/** File name without directories. */
	file: string;
	/** File name without `.gguf` and without a shard suffix. */
	stem: string;
	/** The folder holding the file. */
	folder: string | null;
	/** The folder above that one: the org on the owner's disk layout. */
	org: string | null;
}

export function parseModelPath(path: string | undefined | null): ParsedModelPath | null {
	if (!path) return null;

	const parts = path.split(/[\\/]+/).filter(Boolean);
	const file = parts[parts.length - 1];

	if (!file) return null;

	const stem = file.replace(/\.gguf$/i, '').replace(SHARD_REGEX, '');

	return {
		file,
		folder: parts.length >= 2 ? parts[parts.length - 2] : null,
		org: parts.length >= 3 ? parts[parts.length - 3] : null,
		stem
	};
}

export interface ModelNaming {
	/** File stem without the quant suffix, e.g. `Swift-1.5-Qwen3.8-27B`. */
	name: string;
	/** Quant to show, e.g. `IQ4_XS`. */
	quant: string | null;
	/** The guessed Hugging Face repo, e.g. `ukisai/Swift-1.5-Qwen3.8-27B-GGUF` (to be confirmed). */
	repo: string | null;
}

/**
 * NAME, quant and repo guess. The quant is the ftype's (`Q4_K - Medium` -> `Q4_K`), refined to the
 * file's own suffix when that starts with it (`Q4_K_M`), which says more.
 */
export function modelNaming(
	path: string | undefined | null,
	ftype: string | undefined | null
): ModelNaming | null {
	const parsed = parseModelPath(path);

	if (!parsed) return null;

	const fromFtype = quantFromFtype(ftype);

	let name = parsed.stem;
	let suffix: string | null = null;

	if (fromFtype) {
		const re = new RegExp(`[-_.](${fromFtype.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})$`, 'i');
		const m = re.exec(name);

		if (m) {
			suffix = m[1];
			name = name.slice(0, m.index);
		}
	}

	if (suffix === null) {
		const m = QUANT_SUFFIX_REGEX.exec(name);

		if (m) {
			suffix = m[1];
			name = name.slice(0, m.index);
		}
	}

	let quant = fromFtype;

	if (suffix && (!fromFtype || suffix.toUpperCase().startsWith(fromFtype.toUpperCase()))) {
		quant = suffix;
	}

	name = name.replace(/[-_.]GGUF$/i, '') || parsed.stem;

	return {
		name,
		quant,
		repo: parsed.org ? `${parsed.org}/${name}-GGUF` : null
	};
}
