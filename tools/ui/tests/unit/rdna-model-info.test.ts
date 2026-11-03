import { buildCommit, buildNumber, rdnaCommitUrl } from '$lib/rdna/constants';
import { cardLabelSize, designLabelSize, hardwareLabel, memorySummary } from '$lib/rdna/hardware';
import {
	argValue,
	hasArg,
	isClassifier,
	modelArgs,
	modelNaming,
	parseModelPath,
	presetArgs,
	quantFromFtype,
	runtimeFromDevices,
	runtimeTag
} from '$lib/rdna/model-info';
import { describe, expect, it } from 'vitest';

// argument lists as the production router reports them (2026-11-03), shortened
const SWIFT_ARGS = [
	'C:\\Users\\serry\\source\\ROCmFPX\\build-hip\\bin\\llama-server.exe',
	'--cache-disk',
	'E:\\llama-cache',
	'--ctx-size',
	'98304',
	'--device',
	'ROCm0',
	'--kv-unified',
	'--model',
	'E:\\Models\\ukisai\\Swift-1.5-IQ4_XS\\Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf',
	'--n-gpu-layers',
	'999',
	'--parallel',
	'2'
];
const STARTLUX_ARGS = [
	'llama-server.exe',
	'--classifier-cache',
	'E:\\llama-cache\\.classifier-cache-v1',
	'--classifier-config',
	'E:\\Models\\startlux-models\\StartLux-Decision-v1.1-2B-Instant-BF16-GGUF\\classifier.json',
	'--device',
	'none',
	'--model',
	'E:\\Models\\startlux-models\\StartLux-Decision-v1.1-2B-Instant-BF16-GGUF\\StartLux-Decision-v1.1-2B-Instant-BF16.gguf',
	'--n-gpu-layers',
	'0'
];

describe('arguments', () => {
	it('reads option values in both spellings', () => {
		expect(argValue(SWIFT_ARGS, '--device')).toBe('ROCm0');
		expect(argValue(['--device=Vulkan1'], '--device', '-dev')).toBe('Vulkan1');
		expect(argValue(['-dev', 'CUDA0'], '--device', '-dev')).toBe('CUDA0');
		expect(argValue(SWIFT_ARGS, '--missing')).toBeUndefined();
		expect(argValue(undefined, '--device')).toBeUndefined();
		expect(hasArg(SWIFT_ARGS, '--kv-unified')).toBe(true);
	});

	it('turns a router preset into arguments (unloaded models carry no args)', () => {
		const args = presetArgs(
			'[qwen/qwen3.8-27b]\ndevice = ROCm0\nmodel = E:\\Models\\a\\b\\c-Q8_0.gguf\nkv-unified = true\n'
		);

		expect(argValue(args, '--device')).toBe('ROCm0');
		expect(argValue(args, '--model')).toBe('E:\\Models\\a\\b\\c-Q8_0.gguf');
		expect(modelArgs({ id: 'x', status: { args: [], preset: 'device = Vulkan0' } })).toEqual([
			'--device',
			'Vulkan0'
		]);
		expect(modelArgs({ id: 'x', status: { args: ['--device', 'ROCm0'] } })).toEqual([
			'--device',
			'ROCm0'
		]);
		expect(modelArgs({ id: 'x' })).toEqual([]);
	});
});

describe('runtime and classifier tags', () => {
	it('names the runtime from --device', () => {
		expect(runtimeTag(SWIFT_ARGS)).toBe('ROCm');
		expect(runtimeTag(STARTLUX_ARGS)).toBe('CPU');
		expect(runtimeTag(['--device', 'Vulkan0'])).toBe('Vulkan');
		expect(runtimeTag(['--device', 'CUDA0,CUDA1'])).toBe('CUDA');
		expect(runtimeTag(['--device', 'ROCm0,ROCm1'])).toBe('ROCm');
		expect(runtimeTag(['--device', 'ROCm0,Vulkan1'])).toBe('ROCm + Vulkan');
	});

	it('falls back to --n-gpu-layers 0 = CPU, else unknown', () => {
		expect(runtimeTag(['--n-gpu-layers', '0'])).toBe('CPU');
		expect(runtimeTag(['-ngl', '99'])).toBeNull();
		expect(runtimeTag([])).toBeNull();
	});

	it('names the runtime from /props devices', () => {
		expect(runtimeFromDevices([{ name: 'ROCm0', type: 'gpu' }])).toBe('ROCm');
		expect(runtimeFromDevices([{ name: 'CPU', type: 'cpu' }])).toBe('CPU');
		expect(runtimeFromDevices(undefined)).toBeNull();
	});

	it('tags classifiers by --classifier-head / --classifier-config', () => {
		expect(isClassifier(STARTLUX_ARGS)).toBe(true);
		expect(isClassifier(['--classifier-head', 'x.gguf'])).toBe(true);
		expect(isClassifier(SWIFT_ARGS)).toBe(false);
		expect(isClassifier(['--classifier-cache', 'x'])).toBe(false);
	});
});

describe('name, quant and repo', () => {
	it('parses the model path (Windows and POSIX)', () => {
		expect(
			parseModelPath('E:\\Models\\ukisai\\Swift-1.5-IQ4_XS\\Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf')
		).toEqual({
			file: 'Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf',
			folder: 'Swift-1.5-IQ4_XS',
			org: 'ukisai',
			stem: 'Swift-1.5-Qwen3.8-27B-IQ4_XS'
		});
		expect(parseModelPath('/models/org/repo/Model-Q4_K_M-00001-of-00003.gguf')?.stem).toBe(
			'Model-Q4_K_M'
		);
		expect(parseModelPath('model.gguf')).toEqual({
			file: 'model.gguf',
			folder: null,
			org: null,
			stem: 'model'
		});
		expect(parseModelPath(undefined)).toBeNull();
	});

	it('takes the quant from the ftype before " - "', () => {
		expect(quantFromFtype('IQ4_XS - 4.25 bpw')).toBe('IQ4_XS');
		expect(quantFromFtype('BF16')).toBe('BF16');
		expect(quantFromFtype('Q4_K - Medium')).toBe('Q4_K');
		expect(quantFromFtype(undefined)).toBeNull();
	});

	it('names Swift 1.5 and StartLux as the owner specified', () => {
		expect(
			modelNaming(
				'E:\\Models\\ukisai\\Swift-1.5-IQ4_XS\\Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf',
				'IQ4_XS - 4.25 bpw'
			)
		).toEqual({
			name: 'Swift-1.5-Qwen3.8-27B',
			quant: 'IQ4_XS',
			repo: 'ukisai/Swift-1.5-Qwen3.8-27B-GGUF'
		});
		expect(
			modelNaming(
				'E:\\Models\\startlux-models\\StartLux-Decision-v1.1-2B-Instant-BF16-GGUF\\StartLux-Decision-v1.1-2B-Instant-BF16.gguf',
				'BF16'
			)
		).toEqual({
			name: 'StartLux-Decision-v1.1-2B-Instant',
			quant: 'BF16',
			repo: 'startlux-models/StartLux-Decision-v1.1-2B-Instant-GGUF'
		});
	});

	it('refines a K-quant ftype with the file suffix, strips unknown ftypes by pattern', () => {
		expect(modelNaming('/m/org/x/Qwen3-8B-Q4_K_M.gguf', 'Q4_K - Medium')).toEqual({
			name: 'Qwen3-8B',
			quant: 'Q4_K_M',
			repo: 'org/Qwen3-8B-GGUF'
		});
		expect(modelNaming('/m/unsloth/x/Qwen3-8B-UD-Q4_K_XL.gguf', null)?.name).toBe('Qwen3-8B');
		expect(modelNaming('/m/me/x/Swift-ROCmFP4-FP6.gguf', null)).toEqual({
			name: 'Swift',
			quant: 'ROCmFP4-FP6',
			repo: 'me/Swift-GGUF'
		});
		expect(modelNaming('/m/me/x/plain-model.gguf', 'F16')).toEqual({
			name: 'plain-model',
			quant: 'F16',
			repo: 'me/plain-model-GGUF'
		});
		expect(modelNaming('model.gguf', null)).toEqual({ name: 'model', quant: null, repo: null });
	});
});

describe('build info', () => {
	it('splits build number and commit', () => {
		expect(buildCommit('b12781-707407de1')).toBe('707407de1');
		expect(buildNumber('b12781-707407de1')).toBe('b12781');
		expect(rdnaCommitUrl('b12781-707407de1')).toBe(
			'https://github.com/serrynaimo/RDNA-FAST/commit/707407de1'
		);
		expect(rdnaCommitUrl('unknown')).toBeNull();
		expect(buildNumber(null)).toBeNull();
	});
});

describe('hardware', () => {
	const gpu = (description: string, total = 21458059264, free = 1395864371) => ({
		description,
		memory_free: free,
		memory_total: total,
		name: 'ROCm0',
		type: 'gpu'
	});

	it('counts identical GPUs and leaves the CPU, accelerators and meta devices out', () => {
		expect(hardwareLabel([gpu('AMD Radeon RX 7900 XT')])).toBe('AMD Radeon RX 7900 XT');
		expect(hardwareLabel([gpu('AMD Radeon RX 7900 XTX'), gpu('AMD Radeon RX 7900 XTX')])).toBe(
			'2× AMD Radeon RX 7900 XTX'
		);
		expect(
			hardwareLabel([
				gpu('AMD Radeon RX 7900 XTX'),
				gpu('AMD Radeon RX 7800 XT'),
				{ description: 'AMD Ryzen 9 7900', name: 'CPU', type: 'cpu' },
				{ name: 'BLAS', type: 'accel' },
				{ name: 'Meta', type: 'meta' }
			])
		).toBe('AMD Radeon RX 7900 XTX + AMD Radeon RX 7800 XT');
		expect(
			hardwareLabel([
				{ description: 'AMD Ryzen 9 7900 12-Core Processor', name: 'CPU', type: 'cpu' }
			])
		).toBe('AMD Ryzen 9 7900 12-Core Processor');
		expect(hardwareLabel([])).toBeNull();
	});

	it('sums GPU memory as VRAM and calls a CPU model RAM', () => {
		expect(memorySummary([gpu('a', 20e9, 2e9), gpu('a', 20e9, 4e9)])).toEqual({
			kind: 'VRAM',
			total: 40e9,
			used: 34e9
		});
		expect(
			memorySummary([
				gpu('a', 20e9, 2e9),
				{ memory_free: 9e9, memory_total: 33e9, name: 'CPU', type: 'cpu' }
			])
		).toEqual({ kind: 'VRAM', total: 20e9, used: 18e9 });
		expect(
			memorySummary([{ memory_free: 9e9, memory_total: 33e9, name: 'CPU', type: 'cpu' }])?.kind
		).toBe('RAM');
		expect(
			memorySummary([{ memory_free: 0, memory_total: 0, name: 'ROCm0', type: 'gpu' }])
		).toBeNull();
	});

	it('shrinks the card label so it never runs into the corner logo', () => {
		expect(designLabelSize('AMD RX 7900 XT')).toBe(3.8);
		expect(designLabelSize('AMD Radeon RX 7900 XT')).toBe(3.1);
		expect(designLabelSize('2× AMD Radeon RX 7900 XTX')).toBe(2.6);
		// a short label keeps the design size
		expect(cardLabelSize('AMD RX 7900 XT', 700)).toBe(3.8);
		// a long one shrinks until it fits the 39cqw column with 4% to spare
		const size = cardLabelSize('2× AMD Radeon RX 7900 XTX + AMD Radeon RX 7800 XT', 2600);

		expect(size).toBeLessThan(2.6);
		expect((size * 2600) / 100).toBeLessThanOrEqual(39 * 0.96);
		expect(cardLabelSize('x', null)).toBe(3.8);
	});
});
