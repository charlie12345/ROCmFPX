/**
 * RDNA-FAST bench: results as the owner's scripts record them (agentic_bench.py chat(), edit_gate.py
 * rows, readme_bench.py rows) and the run the page keeps in localStorage.
 */

/** One request of the agentic session, measured as agentic_bench.py chat() measures it. */
export interface SessionStep {
	step: string;
	conv: 'A' | 'B';
	kind: string;
	wall_s: number;
	ttft_s: number | null;
	prompt_tokens: number | null;
	completion_tokens: number | null;
	max_tokens: number;
	error: string | null;
	decode_tps: number | null;
	/** Polls during this request that saw another slot busy. */
	overlap: number;
	srv_prompt_n?: number | null;
	srv_cache_n?: number | null;
	srv_prefill_tps?: number | null;
	srv_decode_tps?: number | null;
	draft_n?: number | null;
	draft_acc?: number | null;
}

export interface SessionResult {
	session_s: number;
	steps: SessionStep[];
}

/** One code edit of edit_gate.py. */
export interface EditRow {
	case: string;
	temp: number;
	prompt_n: number | null;
	gen: number | null;
	decode_tps: number | null;
	draft_n: number;
	draft_acc: number;
	verbatim_lines: number;
	overlap: number;
	error: string | null;
}

/** One call of readme_bench.py chat(). */
export interface ReadmeCall {
	wall_s: number;
	prompt_tok: number | null;
	prefill_tps: number | null;
	gen_tok: number | null;
	decode_tps: number | null;
	draft_n: number;
	draft_acc: number;
	overlap: number;
	error: string | null;
}

/** A row of readme_bench.py: nine decodes summarised. */
export interface ReadmeRow {
	median: number;
	peak: number;
	min: number;
	n: number;
	draft_acceptance: number | null;
}

/** The figures of one run (fair_table.py for the session and edits, readme_bench.py for the rest). */
export type BenchValueKey =
	| 'session'
	| 'decode_4k'
	| 'decode_17k'
	| 'decode_41k'
	| 'prefill_4k'
	| 'prefill_17k'
	| 'prefill_41k'
	| 'edit'
	| 'prose'
	| 'code';

export type BenchValues = Partial<Record<BenchValueKey, number>>;

/** Token counts behind the labels ("4.3k new at 17k depth"). */
export type BenchLabelKey = 'tok_4k' | 'tok_17k' | 'tok_41k' | 'new_17k' | 'new_41k';

export type BenchLabels = Partial<Record<BenchLabelKey, number>>;

export interface BenchRun {
	v: 1;
	/** When the run finished (epoch ms). */
	t: number;
	modelId: string;
	build: string | null;
	nCtx: number | null;
	modelLabel: string;
	/** Other traffic shared the server during the run. */
	overlapped: boolean;
	/** Stopped or failed before the end: it keeps what it measured. */
	incomplete?: boolean;
	/** The run still going (never stored): what it measured so far. */
	inProgress?: boolean;
	/** Steps left out because the model's context cannot hold them, or stopped. */
	skipped: string[];
	sessions: SessionResult[];
	edits: EditRow[];
	prose: ReadmeRow | null;
	code: ReadmeRow | null;
	/*
	 * Runs saved before 2026-11-03 also carry `cold` and cold_* / coldtok_* figures (the cold
	 * prompts, since dropped); nothing reads them.
	 */
	values: BenchValues;
	labels: BenchLabels;
}
