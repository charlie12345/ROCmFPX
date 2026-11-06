/**
 * RDNA-FAST status page: the server's answers (as the API contract describes them) and what the
 * page keeps in localStorage. Every server field is optional: an older server, or one without the
 * NEW endpoints, leaves them out and the page shows a dash.
 */

export interface RdnaModelMeta {
	n_params?: number;
	size?: number;
	ftype?: string;
	n_ctx?: number;
	n_ctx_train?: number;
}

export interface RdnaModelEntry {
	id: string;
	aliases?: string[];
	status?: { value?: string; args?: string[]; preset?: string };
	meta?: RdnaModelMeta | null;
}

export interface RdnaSlot {
	id: number;
	n_ctx?: number;
	is_processing?: boolean;
	priority?: string;
	is_held?: boolean;
	id_task?: number;
	n_prompt_tokens?: number;
	n_prompt_tokens_processed?: number;
	n_prompt_tokens_cache?: number;
	next_token?: { n_decoded?: number }[];
}

export interface RdnaDevice {
	name?: string;
	description?: string;
	type?: string;
	memory_total?: number;
	memory_free?: number;
}

export interface RdnaPromptCacheDisk {
	enabled?: boolean;
	entries?: number;
	bytes?: number;
	limit_bytes?: number;
	tokens?: number;
	saves?: number;
	loads?: number;
	evictions?: number;
	bytes_read?: number;
	bytes_written?: number;
	restore_ms_total?: number;
	restores_timed?: number;
}

export interface RdnaProps {
	role?: string;
	build_info?: string;
	total_slots?: number;
	model_ftype?: string;
	model_path?: string;
	modalities?: { vision?: boolean; audio?: boolean; video?: boolean };
	default_generation_settings?: { n_ctx?: number };
	devices?: RdnaDevice[];
	prompt_cache_disk?: RdnaPromptCacheDisk;
	/** The host's RAM and the model process's share of it (bytes). */
	system_memory?: { memory_total?: number; memory_free?: number; process_resident?: number };
}

export interface CompletionRecord {
	kind: 'completion';
	id: number;
	slot?: number;
	priority?: string;
	status?: string;
	t_arrival_ms: number;
	t_start_ms?: number;
	t_first_ms?: number | null;
	t_end_ms: number;
	n_ctx?: number;
	n_prompt?: number;
	n_cache?: number;
	n_processed?: number;
	prompt_ms?: number;
	n_gen?: number;
	gen_ms?: number;
	draft_n?: number;
	draft_accepted?: number;
}

/** One question of a classifier request (a request may ask several). */
export interface ClassifierQuestion {
	type?: string;
	n_options?: number;
	answer?: string;
	prob?: number;
	cached?: boolean;
}

export interface ClassifierRecord {
	kind: 'classifier';
	id: number;
	priority?: string;
	status?: string;
	type?: string;
	n_options?: number;
	answer?: string;
	prob?: number;
	latency_ms?: number;
	cached?: boolean;
	n_prompt?: number;
	t_arrival_ms: number;
	t_end_ms: number;
	n_questions?: number;
	/** Every question of the request; the top-level fields describe the first. */
	questions?: ClassifierQuestion[];
}

export type RequestRecord = CompletionRecord | ClassifierRecord;

/** One peak per depth bucket: the value and when it was set. */
export interface PeakEntry {
	v: number;
	t: number;
}

export type PeakMap = Record<number, PeakEntry>;

export interface PeaksMemory {
	/** Decode peaks with fast text copies (the "Text copy boost" set). */
	decode: PeakMap;
	/** Decode peaks without fast copies: committed when a request finishes, never lowered. */
	decodeNoCopy: PeakMap;
	prefill: PeakMap;
	decodeResetAt: number;
	prefillResetAt: number;
}

export interface PriorityStat {
	n: number;
	ttftSum: number;
	ttftN: number;
}

export interface PriorityMemory {
	resetAt: number;
	byPriority: Record<string, PriorityStat>;
}

/**
 * SSD cache figures since the last reset: the server's counters run since its start, so the page
 * keeps their values at the reset; reused tokens and prefill time saved add up from /requests.
 */
export interface CacheMemory {
	at: number;
	/** Counters at the reset; null until the page first sees them. */
	counters: RdnaPromptCacheDisk | null;
	/** Prompt tokens served from a cache (slot or SSD) since the reset. */
	reused: number;
	/** Prefill time those tokens would have taken at the measured rates. */
	savedMs: number;
}

export interface ClassifierMemory {
	speedResetAt: number;
	n: number;
	firstT: number | null;
	latencies: number[];
	promptPeak: number | null;
	/** The fastest answer the model computed (ms; cache answers left out), lower is better. */
	fastest: number | null;
	typesResetAt: number;
	types: Record<string, number>;
}
