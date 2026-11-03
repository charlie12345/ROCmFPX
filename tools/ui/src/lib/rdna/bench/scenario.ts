/**
 * RDNA-FAST bench: the prompts of the owner's bench scripts, byte for byte.
 *   agentic_bench.py   the scripted two-agent session (14 requests, 4k / 17k / 41k deep)
 *   edit_gate.py       three code edits that copy the prompt (n-gram drafting)
 *   readme_bench.py    nine prose and nine code answers of 450 tokens (its cold prompts are left out)
 * The long texts come from static/rdna/bench/*.json (scripts/rdna-bench-fixtures.mjs), fetched only
 * when a bench starts.
 */

import { base } from '$app/paths';

export interface AgenticFixture {
	docs: string;
	chunks: string[];
	worker: string;
}

export interface EditFixture {
	code60000: string;
	code120000: string;
}

export interface BenchFixtures {
	agentic: AgenticFixture;
	edit: EditFixture;
}

export async function loadFixtures(signal?: AbortSignal): Promise<BenchFixtures> {
	const get = async <T>(name: string): Promise<T> => {
		const res = await fetch(`${base}/rdna/bench/${name}`, { signal });

		if (!res.ok) throw new Error(`bench fixture ${name}: HTTP ${res.status}`);

		return (await res.json()) as T;
	};
	const [agentic, edit] = await Promise.all([
		get<AgenticFixture>('agentic.json'),
		get<EditFixture>('edit.json')
	]);

	return { agentic, edit };
}

export interface ChatMessage {
	role: 'system' | 'user' | 'assistant';
	content: string;
}

export interface SessionRequest {
	name: string;
	conv: 'A' | 'B';
	kind: 'cold' | 'append' | 'return';
	messages: ChatMessage[];
	maxTokens: number;
}

const LONG = ' Answer with a detailed written analysis of at least 600 words, in prose.';
const ACK = 'Noted. Reading the next file.';

/** The measuring points of fair_table.py. */
export const PICK_4K = 'A1 cold start (4k)';
export const PICK_17K = 'A4 + file (17k)';
export const PICK_41K = 'A12 + file (41k)';

/** A fresh session id, as uuid.uuid4().hex[:12]. */
export function sessionNonce(): string {
	const bytes = new Uint8Array(6);

	crypto.getRandomValues(bytes);

	return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
}

/** agentic_bench.py build(nonce) at --scale 1.0. */
export function buildSession(fx: AgenticFixture, nonce: string): SessionRequest[] {
	const sysA =
		`Session id: ${nonce}\nYou are the orchestrator agent of a software team. You plan work, read files through tools and ` +
		`review code. The reference documentation of the server you maintain follows.\n\n` +
		fx.docs;
	const sysB =
		`Session id: ${nonce}-w\nYou are a worker agent. You receive one source file and one narrowly scoped task from the ` +
		`orchestrator and report back.\n\n[file src/llama-context.cpp]\n` +
		fx.worker;
	const toolMsg = (i: number) =>
		`[tool result: read_file part ${i + 1} of tools/server/server-context.cpp]\n${fx.chunks[i]}\n\n` +
		`Review this part: what does it do and what would you change?${LONG}`;
	const reqs: SessionRequest[] = [];
	const A: ChatMessage[] = [
		{ content: sysA, role: 'system' },
		{
			content:
				'Task: we will review tools/server/server-context.cpp part by part. Before the first file arrives, describe how you will approach the review.' +
				LONG,
			role: 'user'
		}
	];

	reqs.push({ conv: 'A', kind: 'cold', maxTokens: 300, messages: A.slice(), name: PICK_4K });

	for (let i = 0; i < 3; i++) {
		A.push({ content: ACK, role: 'assistant' }, { content: toolMsg(i), role: 'user' });
		reqs.push({
			conv: 'A',
			kind: 'append',
			maxTokens: i === 2 ? 300 : 100,
			messages: A.slice(),
			name: `A${i + 2} + file${i === 2 ? ' (17k)' : ''}`
		});
	}

	const B1: ChatMessage[] = [
		{ content: sysB, role: 'system' },
		{
			content:
				'Task: list every place in this file where the memory of a sequence is modified.' + LONG,
			role: 'user'
		}
	];

	reqs.push({
		conv: 'B',
		kind: 'cold',
		maxTokens: 200,
		messages: B1,
		name: 'B1 worker, cold (10k)'
	});

	A.push(
		{ content: ACK, role: 'assistant' },
		{
			content:
				'The worker is done. Before the next file: which of the parts so far carries the most risk?' +
				LONG,
			role: 'user'
		}
	);
	reqs.push({
		conv: 'A',
		kind: 'return',
		maxTokens: 150,
		messages: A.slice(),
		name: 'A6 back at 17k'
	});

	for (let i = 3; i < 9; i++) {
		A.push({ content: ACK, role: 'assistant' }, { content: toolMsg(i), role: 'user' });
		reqs.push({
			conv: 'A',
			kind: 'append',
			maxTokens: i === 8 ? 300 : 100,
			messages: A.slice(),
			name: `A${i + 4} + file${i === 8 ? ' (41k)' : ''}`
		});
	}

	const B2: ChatMessage[] = [
		B1[0],
		{
			content:
				'Task: list every environment variable this file reads and what each one changes.' + LONG,
			role: 'user'
		}
	];

	reqs.push({
		conv: 'B',
		kind: 'return',
		maxTokens: 200,
		messages: B2,
		name: 'B2 worker again (10k)'
	});

	A.push(
		{ content: ACK, role: 'assistant' },
		{
			content:
				'The worker is done. Summarise the three most important findings of the whole review.' +
				LONG,
			role: 'user'
		}
	);
	reqs.push({
		conv: 'A',
		kind: 'return',
		maxTokens: 150,
		messages: A.slice(),
		name: 'A14 back at 41k'
	});

	return reqs;
}

export interface EditCase {
	name: string;
	temp: number;
	code: string;
	task: string;
}

const RENAME =
	'Return this C++ code complete and unchanged, except: rename the variable `res` to `result` wherever it occurs. Output only the code, no explanation, no code fence.';
const COMMENT =
	'Return this C++ code complete and unchanged, except: add the comment line `// reviewed` as the very first line. Output only the code, no explanation, no code fence.';

/** edit_gate.py CASES: thinking off, 1,000 tokens each. */
export function editCases(fx: EditFixture): EditCase[] {
	return [
		{ code: fx.code60000, name: 'rename', task: RENAME, temp: 0 },
		{ code: fx.code120000, name: 'comment', task: COMMENT, temp: 0 },
		{ code: fx.code60000, name: 'rename1', task: RENAME, temp: 1 }
	];
}

export const EDIT_MAX_TOKENS = 1000;

/** Share of the answer's non-empty lines that are verbatim lines of the input (edit_gate.py). */
export function verbatimShare(code: string, text: string): number {
	const linesIn = new Set(
		code
			.split('\n')
			.map((l) => l.trim())
			.filter(Boolean)
	);
	const out = text
		.split('\n')
		.map((l) => l.trim())
		.filter(Boolean);
	const same = out.filter((l) => linesIn.has(l)).length;

	return Math.round((same / Math.max(1, out.length)) * 100) / 100;
}

/** readme_bench.py PROSE and CODE, 450 tokens each. */
export const PROSE = [
	'Write a 350-word essay on why medieval Baltic amber traders might have preferred overland routes in wet years. Be specific and avoid lists.',
	'Describe, in about 350 words of flowing prose, a plausible failure mode of a hand-built tide clock in a Norwegian fishing village in 1890.',
	'In roughly 350 words, argue for and against teaching bookbinding in a modern secondary-school curriculum, in prose, no bullet points.',
	'In about 350 words of prose, explain how a lighthouse keeper in 1920s Patagonia might have improvised a replacement for a cracked Fresnel prism.',
	'Write roughly 350 words of prose on the economics of a travelling knife-grinder in 1930s Lisbon. No lists.',
	'In around 350 words, describe the social consequences of a village in the Carpathians adopting a shared bread oven in 1780. Prose only.',
	"Write 350 words of prose on why a Victorian botanist might distrust a colleague's pressed-fern collection from Tasmania.",
	'In about 350 words of prose, recount how a canal lock-keeper in 1840s Languedoc might settle a dispute between two barge crews.',
	'Write about 350 words of prose on how a 1950s Finnish radio repairman might explain static to a worried customer. No lists.'
];

export const CODE = [
	"Write a Python function that parses an ISO 8601 duration string like 'P3DT4H5M' into a datetime.timedelta, with docstring and a few asserts. Code only.",
	'Write a TypeScript function that debounces an async function and returns the latest promise, with types and a short usage example. Code only.',
	'Write a Python class implementing an LRU cache with a maximum byte size (not item count), with get/put and eviction. Code only.',
	'Write a Rust function that reads a CSV file and returns the sum of a named numeric column, with error handling via Result. Code only.',
	'Write a Go HTTP handler that streams a large file with Range request support, including the headers. Code only.',
	'Write a Python script that walks a directory tree and prints duplicate files by SHA-256, grouping paths. Code only.',
	'Write a SQL schema plus three queries for a library lending system (books, members, loans, overdue report). Code only.',
	'Write a JavaScript function that converts a nested object into dot-notation key paths and back, with tests. Code only.',
	'Write a C function that implements a ring buffer for bytes with push, pop and peek, plus a tiny test main. Code only.'
];

export const README_MAX_TOKENS = 450;
/**
 * Prompt tokens of each part on the Qwen3.8 tokenizer (measured 2026-11-03), to skip what a model's
 * context cannot hold. Other tokenizers differ, so the check keeps 10% to spare.
 */
export const TOKENS_SESSION_MAX = 41107;
export const TOKENS_EDIT_MAX = 2285;
export const TOKEN_MARGIN = 1.1;

export function fits(nCtx: number | null, promptTokens: number, maxTokens: number): boolean {
	if (!nCtx || nCtx <= 0) return true;

	return promptTokens * TOKEN_MARGIN + maxTokens <= nCtx;
}
