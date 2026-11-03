#!/usr/bin/env node
// RDNA-FAST: writes the prompt fixtures of the browser bench (src/lib/rdna/bench) into static/rdna/bench/.
//
// The bench in the status page is a browser port of the owner's bench scripts; these fixtures make its long
// prompts byte-identical to theirs (readme_bench.py's short prompts live in src/lib/rdna/bench/scenario.ts):
//   agentic.json   agentic_bench.py --scale 1.0: scenario.json as written by make_scenario.py (docs, nine source
//                  chunks, worker slice, cut to exact token counts so the session is 4k / 17k / 41k tokens deep)
//   edit.json      edit_gate.py: two 9,600-character slices (whole lines) of tools/server/server-context.cpp at
//                  commit 3e500ba13, starting after char 60,000 and 120,000
//
// Usage (from tools/ui):
//   node scripts/rdna-bench-fixtures.mjs --scenario <scenario.json>
// The edit slices are read from this repository with `git show 3e500ba13:...`.

import { execFileSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const uiDir = resolve(here, '..');
const repoDir = resolve(uiDir, '..', '..');
const outDir = resolve(uiDir, 'static', 'rdna', 'bench');

function arg(name) {
	const i = process.argv.indexOf(name);

	return i > 0 ? process.argv[i + 1] : undefined;
}

const scenarioPath = arg('--scenario');

if (!scenarioPath) {
	console.error('usage: node scripts/rdna-bench-fixtures.mjs --scenario <scenario.json>');
	process.exit(2);
}

mkdirSync(outDir, { recursive: true });

// 1. agentic session: the scenario as is (docs, chunks[9], worker)
const scenario = JSON.parse(readFileSync(scenarioPath, 'utf-8'));

if (
	!scenario.docs ||
	!Array.isArray(scenario.chunks) ||
	scenario.chunks.length !== 9 ||
	!scenario.worker
) {
	console.error('scenario.json: expected { docs, chunks[9], worker }');
	process.exit(1);
}

writeFileSync(
	resolve(outDir, 'agentic.json'),
	JSON.stringify({
		chunks: scenario.chunks,
		docs: scenario.docs,
		source: 'agentic_bench.py scenario.json (make_scenario.py, texts of commit 3e500ba13)',
		worker: scenario.worker
	})
);

// 2. code edits: edit_gate.py chunk(start, 9600), whole lines only
const src = execFileSync(
	'git',
	['-C', repoDir, 'show', '3e500ba13:tools/server/server-context.cpp'],
	{
		maxBuffer: 64 * 1024 * 1024
	}
).toString('utf-8');

function chunk(start, nChars = 9600) {
	const a = src.indexOf('\n', start) + 1;
	const b = src.indexOf('\n', a + nChars);

	return src.slice(a, b);
}

writeFileSync(
	resolve(outDir, 'edit.json'),
	JSON.stringify({
		code60000: chunk(60000),
		code120000: chunk(120000),
		source: 'edit_gate.py: tools/server/server-context.cpp at commit 3e500ba13'
	})
);

for (const f of ['agentic.json', 'edit.json']) {
	console.log(`wrote static/rdna/bench/${f} (${readFileSync(resolve(outDir, f)).length} bytes)`);
}
