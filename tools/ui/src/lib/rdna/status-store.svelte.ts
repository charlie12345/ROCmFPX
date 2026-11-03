/**
 * RDNA-FAST status page store: polls the server (/slots and /requests every second, /props every
 * ten, /v1/models every five), keeps what the page remembers in localStorage per model id, and
 * derives every tile's figures. One instance per mounted status page.
 *
 *   /v1/models            model tabs (name, runtime and classifier tags, loaded or not)
 *   /props                build pill, header chips, hardware label, VRAM, SSD cache counters
 *   /props?model=         the same per model in router mode
 *   /slots?model=         slots tile, live lines of Decode / Prefill, running rows of Requests
 *   /requests?model=      finished requests: history, peaks, priority, SSD reuse, classifier tiles
 *
 * Per-model GETs carry autoload=false and only go to loaded models, so the page never loads one.
 */

import { modelQuery, rdnaGet } from './api';
import { BENCH_CHART_KEYS, bestOfLastTwo, sortedRuns } from './bench/metrics';
import { benchRunner, loadBenchRuns } from './bench/runner.svelte';
import type { BenchRun } from './bench/types';
import { addToCache, avgRestoreMs, cacheCountersWentDown, cacheDelta } from './cache';
import { type CardFigures, cardFigures } from './card';
import {
	HISTORY_CAP,
	POLL_MISSING_MS,
	POLL_MODELS_MS,
	POLL_PROPS_MS,
	POLL_REQUESTS_MS,
	POLL_REQUESTS_OTHER_MS,
	POLL_SLOTS_MS
} from './constants';
import { bucketIndex, type DepthBucket, depthBuckets } from './depth';
import { hardwareLabel, memorySummary } from './hardware';
import { type HfRepoInfo, lookupAvatar, lookupRepo, splitRepo } from './hf';
import {
	addToClassifier,
	addToPriority,
	isFastCopy,
	mergeHistory,
	normalizeRecord,
	type PeakCandidate,
	peakCandidates,
	updatePeaks
} from './history';
import { LiveRates, type SlotLive } from './live';
import {
	isClassifier,
	modelArgs,
	type ModelNaming,
	modelNaming,
	modelPathOf,
	runtimeFromDevices,
	runtimeTag,
	slotCountOf
} from './model-info';
import { RecordGlow } from './record-glow.svelte';
import { type SlotMemory, updateSlotMemory } from './slot-memory';
import {
	emptyCache,
	emptyClassifier,
	emptyPeaks,
	emptyPriority,
	loadCache,
	loadClassifier,
	loadCopyBoost,
	loadHardwareLabel,
	loadHistory,
	loadPeaks,
	loadPriority,
	loadSeenHardware,
	loadUiState,
	saveSeenHardware,
	saveUiState,
	writePart
} from './storage';
import type {
	CacheMemory,
	ClassifierMemory,
	PeakMap,
	PeaksMemory,
	PriorityMemory,
	RdnaModelEntry,
	RdnaProps,
	RdnaSlot,
	RequestRecord
} from './types';

export interface ModelView {
	id: string;
	entry: RdnaModelEntry;
	args: string[];
	loaded: boolean;
	/**
	 * Loading: the router says "loading", or the model is listed loaded but its /props or /slots
	 * still answers with a server error while it starts. The page shows no text for it: a spinner
	 * on the tab and pulsing slot dots.
	 */
	loading: boolean;
	status: string;
	runtime: string | null;
	classifier: boolean;
	naming: ModelNaming | null;
	/** Tab label: the model's NAME (with its quant when two tabs would read the same). */
	label: string;
}

export interface HfView {
	repo: HfRepoInfo | null;
	avatar: string | null;
}

export interface DepthRow {
	hi: number;
	label: string;
	peak: number | null;
	live: number | null;
	record: boolean;
}

export interface RunningRow {
	key: string;
	slot: number;
	t: number;
	priority: string;
	phase: SlotLive['phase'];
	ctx: number;
	processed: number;
	cached: number;
	decoded: number;
	prefillTps: number | null;
	decodeTps: number | null;
}

/** A running request's readings wait this long for its /requests record, then are let go. */
const PENDING_NO_COPY_MS = 30 * 60_000;

function isLoadedStatus(v: string | undefined): boolean {
	return v === undefined || v === 'loaded';
}

/** A server error while the model starts (llama-server answers 503 until the model is loaded). */
function isStartingStatus(status: number): boolean {
	return status >= 500 && status < 600;
}

export class StatusStore {
	auth = $state<'unknown' | 'ok' | 'needed'>('unknown');
	benchRuns = $state.raw<BenchRun[]>([]);
	selectedId = $state<string | null>(null);
	/**
	 * The runs the main tile shows: the stored ones plus the run in progress on this model (what it
	 * measured so far), which counts as the newest.
	 */
	benchView: BenchRun[] = $derived.by(() => {
		const partial = benchRunner.partial;
		const own = partial && partial.modelId === this.selectedId ? [partial] : [];

		return sortedRuns([...this.benchRuns, ...own]);
	});
	nCtx: number | null = $derived(this.nCtxOf(this.selectedId));
	buckets: DepthBucket[] = $derived(depthBuckets(this.nCtx));
	rootProps = $state.raw<RdnaProps | null>(null);
	props = $state.raw<Record<string, RdnaProps>>({});
	selectedProps: RdnaProps | null = $derived(
		this.selectedId ? (this.props[this.selectedId] ?? null) : null
	);
	buildInfo: string | null = $derived(
		this.rootProps?.build_info ?? this.selectedProps?.build_info ?? null
	);
	cache = $state.raw<CacheMemory>(emptyCache());
	cacheView = $derived.by(() => {
		const disk = this.selectedProps?.prompt_cache_disk;
		const delta = cacheDelta(disk, this.cache.counters);

		return {
			available: disk !== undefined,
			avgRestoreMs: avgRestoreMs(delta),
			delta,
			disk: disk ?? null,
			enabled: disk ? disk.enabled !== false : false,
			reused: this.cache.reused,
			savedMs: this.cache.savedMs,
			since: this.cache.at
		};
	});
	/** The speed card: bench runs only, the best of the last two (the running one included). */
	card: CardFigures = $derived.by(() => cardFigures(bestOfLastTwo(this.benchView).values));
	classifier = $state.raw<ClassifierMemory>(emptyClassifier());
	/**
	 * The Decode tile's "Text copy boost" (per model, off by default): on shows the decode peaks with
	 * fast text copies (the bar follows the white line on every new record), off the ones without
	 * (a request's best reading is committed when it finishes and is no fast copy). Both sets are
	 * kept, so the toggle switches at once.
	 */
	copyBoost = $state(false);
	peaks = $state.raw<PeaksMemory>(emptyPeaks());

	decodeRows: DepthRow[] = $derived(
		this.depthRows(this.copyBoost ? this.peaks.decode : this.peaks.decodeNoCopy, 'decode')
	);
	entries = $state.raw<RdnaModelEntry[]>([]);
	/** The record glow of every speed figure (record-glow.svelte.ts). */
	glow = new RecordGlow();
	hwOverride = $state<string | null>(null);
	/** The GPU name /props last reported for the model in this tab (storage.ts loadSeenHardware). */
	hwSeen = $state<string | null>(null);
	hardware: string | null = $derived(
		this.hwOverride ??
			(this.selectedProps ? hardwareLabel(this.selectedProps.devices) : this.hwSeen)
	);
	/**
	 * The label is settled: the owner named it, /props answered (with or without devices), or this
	 * tab saw the model's GPU before.
	 */
	hardwareKnown: boolean = $derived(
		this.hwOverride !== null || this.selectedProps !== null || this.hwSeen !== null
	);
	hf = $state.raw<Record<string, HfView>>({});
	hfView: HfView | null = $derived(this.selectedId ? (this.hf[this.selectedId] ?? null) : null);
	history = $state.raw<RequestRecord[]>([]);
	/** Each model's last /slots answer (page session), shown idle while the model loads again. */
	knownSlots = $state.raw<Record<string, RdnaSlot[]>>({});
	live = $state.raw<SlotLive[]>([]);

	memory = $derived(memorySummary(this.selectedProps?.devices));
	/** Models listed loaded whose /props or /slots answers with a server error: still starting. */
	starting = $state.raw<Record<string, boolean>>({});
	models: ModelView[] = $derived.by(() => {
		const views = this.entries.map((e): ModelView => {
			const args = modelArgs(e);
			const props = this.props[e.id];
			const path = modelPathOf(args) ?? props?.model_path;
			const naming = modelNaming(path, e.meta?.ftype ?? props?.model_ftype);

			return {
				args,
				classifier: isClassifier(args),
				entry: e,
				id: e.id,
				label: naming?.name || e.id,
				loaded: isLoadedStatus(e.status?.value),
				loading:
					e.status?.value === 'loading' ||
					(isLoadedStatus(e.status?.value) && !!this.starting[e.id]),
				naming,
				runtime: runtimeTag(args) ?? runtimeFromDevices(props?.devices),
				status: e.status?.value ?? 'loaded'
			};
		});
		// eslint-disable-next-line svelte/prefer-svelte-reactivity -- local to this computation
		const counts = new Map<string, number>();

		for (const v of views) counts.set(v.label, (counts.get(v.label) ?? 0) + 1);

		return views.map((v) =>
			(counts.get(v.label) ?? 0) > 1 && v.naming?.quant
				? { ...v, label: `${v.label} ${v.naming.quant}` }
				: v
		);
	});
	modelsLoaded = $state(false);
	now = $state(Date.now());
	online = $state<boolean | null>(null);
	prefillRows: DepthRow[] = $derived(this.depthRows(this.peaks.prefill, 'prefill'));

	// ---------------------------------------------------------------- derived

	prio = $state.raw<PriorityMemory>(emptyPriority());

	requestsMissing = $state.raw<Record<string, boolean>>({});

	rootKnown = $state(false);

	routerMode = $state(false);
	runningSeen = $state.raw<Record<string, number>>({});
	slots = $state.raw<RdnaSlot[] | null>(null);
	runningRows: RunningRow[] = $derived.by(() => {
		const slots = this.slots ?? [];

		return this.live
			.map((l, i) => ({ l, s: slots[i] }))
			.filter(({ l, s }) => s && l.phase !== 'idle')
			.map(({ l, s }) => {
				const key = `${s.id}:${s.id_task ?? ''}`;

				return {
					cached: l.cached,
					ctx: l.ctx,
					decoded: l.decoded,
					decodeTps: l.decodeTps,
					key,
					phase: l.phase,
					prefillTps: l.prefillTps,
					priority: s.priority ?? 'default',
					processed: l.processed,
					slot: s.id,
					t: this.runningSeen[key] ?? this.now
				};
			});
	});
	runsNewestFirst: BenchRun[] = $derived(this.benchView);

	selected: ModelView | null = $derived(this.models.find((m) => m.id === this.selectedId) ?? null);

	/** A shared KV pool: several slots that each see the whole context. */
	sharedPool: { used: number; total: number } | null = $derived.by(() => {
		const slots = this.slots;
		const total = this.nCtx;

		if (!slots || slots.length < 2 || !total) return null;

		const unified =
			this.selected?.args.includes('--kv-unified') || this.selected?.args.includes('-kvu');

		if (!unified && !slots.every((s) => s.n_ctx === total)) return null;

		return { total, used: slots.reduce((s, x) => s + (x.n_prompt_tokens ?? 0), 0) };
	});

	/**
	 * What the Slots tile shows: the live slots; while the model loads, its last known slots (idle)
	 * or one empty row per slot it will have (its -np, else /props total_slots, else one).
	 */
	shownSlots: RdnaSlot[] | null = $derived.by(() => {
		const m = this.selected;

		if (!m?.loading) return this.slots;

		const known = this.knownSlots[m.id];

		if (known && known.length > 0) return known.map((s) => ({ ...s, is_processing: false }));

		const n = slotCountOf(m.args) ?? this.props[m.id]?.total_slots ?? 1;

		return Array.from({ length: n }, (_, id): RdnaSlot => ({ id }));
	});

	/** Context of one slot (what one request can hold). */
	slotCtx: number | null = $derived(
		this.slots && this.slots.length > 0
			? Math.min(...this.slots.map((s) => s.n_ctx ?? Infinity).filter((n) => Number.isFinite(n)))
			: this.nCtx
	);

	/** Each slot's latest request, kept after it finished (page session only). */
	slotMemory = $state.raw<Record<number, SlotMemory>>({});

	totalSlots: number | null = $derived(
		this.selectedProps?.total_slots ?? (this.slots ? this.slots.length : null)
	);

	private alive = false;

	// eslint-disable-next-line svelte/prefer-svelte-reactivity -- bookkeeping, not state
	private hfRequested = new Set<string>();

	// eslint-disable-next-line svelte/prefer-svelte-reactivity -- bookkeeping, not state
	private ingestedOnce = new Set<string>();

	private liveRates = new LiveRates();

	// eslint-disable-next-line svelte/prefer-svelte-reactivity -- bookkeeping, not state
	private missingCheckedAt = new Map<string, number>();

	/**
	 * The best per-poll decode reading of each running request (by task id, per depth bucket), for
	 * the set without fast copies: committed when the request's record shows it was no fast copy.
	 */
	// eslint-disable-next-line svelte/prefer-svelte-reactivity -- bookkeeping, not state
	private pendingNoCopy = new Map<string, { at: number; best: PeakMap }>();

	private requestsTick = 0;

	private tick: ReturnType<typeof setInterval> | null = null;

	// eslint-disable-next-line svelte/prefer-svelte-reactivity -- bookkeeping, not state
	private timers = new Map<string, ReturnType<typeof setTimeout>>();

	// ---------------------------------------------------------------- lifecycle

	/** Merge one model's fetched records into what the page keeps, and update every figure. */
	ingest(id: string, records: RequestRecord[]): void {
		const isSel = id === this.selectedId;
		const history = isSel ? this.history : loadHistory(id);
		const merged = mergeHistory(history, records, HISTORY_CAP);

		if (merged.added.length === 0) return;

		const added = merged.added;
		const peaks = isSel ? this.peaks : loadPeaks(id);
		const prio = isSel ? this.prio : loadPriority(id);
		const cache = isSel ? this.cache : loadCache(id);
		const cls = isSel ? this.classifier : loadClassifier(id);
		const buckets = depthBuckets(this.nCtxOf(id));
		const cands = peakCandidates(added, buckets, peaks.decodeResetAt, peaks.prefillResetAt);
		const dec = updatePeaks(peaks.decode, cands.decode);
		const noCopy = updatePeaks(peaks.decodeNoCopy, [
			...cands.decodeNoCopy,
			...(isSel ? this.finishedReadings(added, peaks.decodeResetAt) : [])
		]);
		const pre = updatePeaks(peaks.prefill, cands.prefill);
		const nextPeaks: PeaksMemory = {
			...peaks,
			decode: dec.peaks,
			decodeNoCopy: noCopy.peaks,
			prefill: pre.peaks
		};
		const rateAt = (depth: number): number | null => {
			const b = bucketIndex(depth, buckets);
			const own = b >= 0 ? pre.peaks[buckets[b].hi]?.v : undefined;

			if (own) return own;

			const all = Object.values(pre.peaks).map((p) => p.v);

			return all.length > 0 ? Math.max(...all) : null;
		};
		const nextCache = addToCache(cache, added, rateAt);
		const nextPrio = addToPriority(prio, added);
		const nextCls = addToClassifier(cls, added);

		writePart(id, 'history', merged.history);

		if (dec.records.length || noCopy.records.length || pre.records.length) {
			writePart(id, 'peaks', nextPeaks);
		}

		if (nextCache !== cache) writePart(id, 'cache', nextCache);

		if (nextPrio !== prio) writePart(id, 'prio', nextPrio);

		if (nextCls !== cls) writePart(id, 'classifier', nextCls);

		if (isSel) {
			this.history = merged.history;
			this.peaks = nextPeaks;
			this.prio = nextPrio;
			this.cache = nextCache;
			this.classifier = nextCls;

			// the first answer of this page view brings history: what it shows is no record
			if (this.ingestedOnce.has(id)) this.observeRecords();
			else this.seedRecords();
		}

		this.ingestedOnce.add(id);
	}

	/** Feed every speed figure to the glow: new bests glow (the bench's figures as runs fill in). */
	observeRecords(): void {
		const now = Date.now();

		for (const [key, value, lower] of this.recordFigures())
			this.glow.observe(key, value, lower, now);
	}

	reloadBench(): void {
		if (!this.selectedId) return;

		this.benchRuns = loadBenchRuns(this.selectedId);
	}

	resetCache(): void {
		this.cache = {
			at: Date.now(),
			counters: this.selectedProps?.prompt_cache_disk ?? null,
			reused: 0,
			savedMs: 0
		};
		this.persist('cache', this.cache);
	}

	resetClassifierSpeed(): void {
		this.classifier = {
			...this.classifier,
			fastest: null,
			firstT: null,
			latencies: [],
			n: 0,
			promptPeak: null,
			speedResetAt: Date.now()
		};
		this.persist('classifier', this.classifier);
		this.glow.forget('cls:');
	}

	// ---------------------------------------------------------------- resets

	/** Reset clears both decode sets (with and without fast copies). */
	resetDecode(): void {
		this.peaks = { ...this.peaks, decode: {}, decodeNoCopy: {}, decodeResetAt: Date.now() };
		this.persist('peaks', this.peaks);
		this.pendingNoCopy.clear();
		this.glow.forget('decode');
	}

	resetPrefill(): void {
		this.peaks = { ...this.peaks, prefill: {}, prefillResetAt: Date.now() };
		this.persist('peaks', this.peaks);
		this.glow.forget('prefill:');
	}

	resetPriority(): void {
		this.prio = emptyPriority(Date.now());
		this.persist('prio', this.prio);
	}

	resetQuestionTypes(): void {
		this.classifier = { ...this.classifier, types: {}, typesResetAt: Date.now() };
		this.persist('classifier', this.classifier);
	}

	/** The key was saved: poll again at once. */
	retryAuth(): void {
		this.auth = 'unknown';

		for (const name of ['models', 'root', 'props', 'slots', 'requests']) this.kick(name);
	}

	select(id: string): void {
		if (id === this.selectedId) return;

		this.selectedId = id;
		saveUiState({ selectedModel: id });
		this.loadMemory(id);
		this.liveRates.reset();
		this.slots = null;
		this.live = [];
		this.runningSeen = {};
		this.slotMemory = {};
		this.pendingNoCopy.clear();
		this.kick('props');
		this.kick('slots');
		this.kick('requests');
	}

	/** Show the decode peaks with fast text copies (on) or without (off); kept per model. */
	setCopyBoost(on: boolean): void {
		this.copyBoost = on;
		this.persist('copyBoost', on);
	}

	setHardwareLabel(label: string | null): void {
		const v = label && label.trim() ? label.trim() : null;

		this.hwOverride = v;
		this.persist('hw', v);
	}

	// ---------------------------------------------------------------- polling

	start(): void {
		if (this.alive) return;

		this.alive = true;

		const ui = loadUiState();

		if (ui.selectedModel) this.selectedId = ui.selectedModel;

		if (this.selectedId) this.loadMemory(this.selectedId);

		this.loop('models', () => this.pollModels(), POLL_MODELS_MS);
		this.loop('root', () => this.pollRoot(), POLL_PROPS_MS);
		this.loop('props', () => this.pollProps(), POLL_PROPS_MS);
		this.loop('slots', () => this.pollSlots(), POLL_SLOTS_MS);
		this.loop('requests', () => this.pollRequests(), POLL_REQUESTS_MS);
		this.tick = setInterval(() => {
			this.now = Date.now();
			this.glow.prune(this.now);
		}, 1000);
	}

	stop(): void {
		this.alive = false;

		for (const t of this.timers.values()) clearTimeout(t);

		this.timers.clear();

		if (this.tick) clearInterval(this.tick);

		this.tick = null;
	}

	private depthRows(peaks: PeakMap, kind: 'decode' | 'prefill'): DepthRow[] {
		const buckets = this.buckets;
		const live: Record<number, number> = {};

		for (const l of this.live) {
			const v = kind === 'decode' ? l.decodeTps : l.prefillTps;

			if (v === null) continue;

			const b = bucketIndex(l.ctx, buckets);

			if (b < 0) continue;

			const hi = buckets[b].hi;

			live[hi] = Math.max(live[hi] ?? 0, v);
		}

		return buckets.map((b) => ({
			hi: b.hi,
			label: b.label,
			live: live[b.hi] ?? null,
			peak: peaks[b.hi]?.v ?? null,
			record: this.glow.on(this.glowKey(kind, b.hi), this.now)
		}));
	}

	/**
	 * Finished requests' best per-poll decode readings, for the set without fast copies: committed
	 * when the record shows no fast copy (a fast copy's readings are dropped).
	 */
	private finishedReadings(records: readonly RequestRecord[], resetAt: number): PeakCandidate[] {
		const out: PeakCandidate[] = [];

		for (const r of records) {
			if (r.kind !== 'completion') continue;

			const key = String(r.id);
			const p = this.pendingNoCopy.get(key);

			if (!p) continue;

			this.pendingNoCopy.delete(key);

			if (r.status === 'error' || isFastCopy(r)) continue;

			for (const [hi, e] of Object.entries(p.best)) {
				if (e.t > resetAt) out.push({ bucket: Number(hi), t: e.t, value: e.v });
			}
		}

		return out;
	}

	/** A depth row's glow key; the decode set without fast copies has its own. */
	private glowKey(kind: 'decode' | 'prefill', hi: number): string {
		return kind === 'decode' && !this.copyBoost ? `decode-nc:${hi}` : `${kind}:${hi}`;
	}

	private kick(name: string): void {
		if (!this.alive) return;

		const t = this.timers.get(name);

		if (t === undefined) return; // running right now; it schedules itself

		clearTimeout(t);

		const fns: Record<string, [() => Promise<number | void>, number]> = {
			models: [() => this.pollModels(), POLL_MODELS_MS],
			props: [() => this.pollProps(), POLL_PROPS_MS],
			requests: [() => this.pollRequests(), POLL_REQUESTS_MS],
			root: [() => this.pollRoot(), POLL_PROPS_MS],
			slots: [() => this.pollSlots(), POLL_SLOTS_MS]
		};

		this.timers.delete(name);
		this.loop(name, fns[name][0], fns[name][1]);
	}

	private loadMemory(id: string): void {
		this.history = loadHistory(id);
		this.peaks = loadPeaks(id);
		this.copyBoost = loadCopyBoost(id);
		this.prio = loadPriority(id);
		this.classifier = loadClassifier(id);
		this.hwOverride = loadHardwareLabel(id);
		this.hwSeen = loadSeenHardware(id);
		this.benchRuns = loadBenchRuns(id);

		const cache = loadCache(id);

		this.cache = cache;
		this.glow.clear();
		this.seedRecords();
	}

	private lookupHf(): void {
		for (const m of this.models) {
			const repo = m.naming?.repo;

			if (!repo || this.hfRequested.has(`${m.id}|${repo}`)) continue;

			this.hfRequested.add(`${m.id}|${repo}`);

			void (async () => {
				const info = await lookupRepo(repo);
				const root = info?.exists ? info.root : null;
				const org = root ? splitRepo(root).org : splitRepo(repo).org;
				const avatar = org ? await lookupAvatar(org) : null;

				this.hf = {
					...this.hf,
					[m.id]: { avatar, repo: info }
				};
			})();
		}
	}

	private loop(name: string, fn: () => Promise<number | void>, interval: number): void {
		const run = async () => {
			this.timers.delete(name);

			if (!this.alive) return;

			let next = interval;

			if (this.auth !== 'needed') {
				try {
					const r = await fn();

					if (typeof r === 'number') next = r;
				} catch (e) {
					console.warn(`[rdna] ${name} poll failed`, e);
				}
			}

			if (!this.alive || this.timers.has(name)) return;

			this.timers.set(name, setTimeout(run, next));
		};

		this.timers.set(name, setTimeout(run, 0));
	}

	private nCtxOf(id: string | null): number | null {
		if (!id) return null;

		const entry = this.entries.find((e) => e.id === id);

		return (
			(entry?.meta?.n_ctx ??
				this.props[id]?.default_generation_settings?.n_ctx ??
				(id === this.selectedId && this.slots?.[0]?.n_ctx) ??
				null) ||
			null
		);
	}

	private persist(
		part: 'peaks' | 'cache' | 'prio' | 'classifier' | 'hw' | 'history' | 'copyBoost',
		value: unknown
	): void {
		if (this.selectedId) writePart(this.selectedId, part, value);
	}

	// ---------------------------------------------------------------- memory

	private async pollModels(): Promise<void> {
		const res = await rdnaGet<{ data?: RdnaModelEntry[] }>('/v1/models');

		if (!this.seen(res.status) || !res.data) return;

		const list = Array.isArray(res.data.data)
			? res.data.data.filter((e) => e && typeof e.id === 'string')
			: [];

		this.entries = list;
		this.modelsLoaded = true;

		if (!this.selectedId || !list.some((e) => e.id === this.selectedId)) {
			const views = this.models;
			const pick =
				views.find((m) => m.loaded && !m.classifier) ?? views.find((m) => m.loaded) ?? views[0];

			if (pick) {
				this.selectedId = null;
				this.select(pick.id);
			}
		}

		this.lookupHf();
	}

	private async pollProps(): Promise<void> {
		if (!this.rootKnown || !this.routerMode) return;

		const m = this.selected;

		if (!m || !m.loaded) return;

		const res = await rdnaGet<RdnaProps>('/props', modelQuery(m.id, true));

		if (isStartingStatus(res.status)) this.setStarting(m.id, true);

		if (!this.seen(res.status) || !res.data) return;

		this.setStarting(m.id, false);
		this.setProps(m.id, res.data);
	}

	private async pollRequests(): Promise<void> {
		if (!this.rootKnown) return;

		this.requestsTick++;

		const others = this.requestsTick % Math.round(POLL_REQUESTS_OTHER_MS / POLL_REQUESTS_MS) === 0;
		const targets = this.models.filter((m) => m.loaded && (m.id === this.selectedId || others));

		for (const m of targets) {
			if (this.requestsMissing[m.id]) {
				const last = this.missingCheckedAt.get(m.id) ?? 0;

				if (Date.now() - last < POLL_MISSING_MS) continue;
			}

			const res = await rdnaGet<{ requests?: unknown[] }>(
				'/requests',
				modelQuery(m.id, this.routerMode)
			);

			if (res.status === 404 || res.status === 501) {
				this.missingCheckedAt.set(m.id, Date.now());

				if (!this.requestsMissing[m.id]) {
					this.requestsMissing = { ...this.requestsMissing, [m.id]: true };
				}

				continue;
			}

			if (!this.seen(res.status) || !res.data) continue;

			if (this.requestsMissing[m.id]) {
				const next = { ...this.requestsMissing };

				delete next[m.id];
				this.requestsMissing = next;
			}

			const raw = Array.isArray(res.data.requests) ? res.data.requests : [];

			this.ingest(
				m.id,
				raw.map(normalizeRecord).filter((r): r is RequestRecord => r !== null)
			);
		}
	}

	private async pollRoot(): Promise<void> {
		const res = await rdnaGet<RdnaProps>('/props');

		if (isStartingStatus(res.status) && this.rootKnown && !this.routerMode) {
			const id = this.selectedId ?? this.entries[0]?.id;

			if (id) this.setStarting(id, true);
		}

		if (!this.seen(res.status) || !res.data) return;

		this.rootProps = res.data;
		this.routerMode = res.data.role === 'router';
		this.rootKnown = true;

		if (!this.routerMode) {
			// a single-model server: its /props is the model's
			const id = this.selectedId ?? this.entries[0]?.id;

			if (id) {
				this.setStarting(id, false);
				this.setProps(id, res.data);
			}
		}
	}

	private async pollSlots(): Promise<void> {
		if (!this.rootKnown) return;

		if (typeof document !== 'undefined' && document.hidden) return;

		const m = this.selected;

		if (!m || !m.loaded) {
			this.slots = null;
			this.live = [];

			return;
		}

		const id = m.id;
		const res = await rdnaGet<RdnaSlot[]>('/slots', modelQuery(id, this.routerMode));

		if (id !== this.selectedId) return;

		if (!this.seen(res.status) || !Array.isArray(res.data)) {
			if (isStartingStatus(res.status)) this.setStarting(id, true);

			if (res.status !== 0) {
				this.slots = null;
				this.live = [];
			}

			return;
		}

		this.setStarting(id, false);

		const t = Date.now();
		const slots = res.data;
		const live = this.liveRates.update(slots, t);
		const seen: Record<string, number> = {};

		slots.forEach((s, i) => {
			if (live[i].phase === 'idle') return;

			const key = `${s.id}:${s.id_task ?? ''}`;

			seen[key] = this.runningSeen[key] ?? t;
		});

		this.slots = slots;
		this.knownSlots = { ...this.knownSlots, [id]: slots };
		this.live = live;
		this.runningSeen = seen;
		this.slotMemory = updateSlotMemory(this.slotMemory, slots, live, t);
		this.raisePeaksFromLive(live, t);
	}

	/**
	 * The simple peak rule (owner, 2026-11-03): on every poll, a row whose live value (the white
	 * line) is above its peak takes exactly that value as its new peak, and the record glow plays;
	 * otherwise the bar stays where it is. The bucket is the request's current context depth.
	 */
	private raisePeaksFromLive(live: SlotLive[], t: number): void {
		const buckets = this.buckets;

		let decode = this.peaks.decode;
		let prefill = this.peaks.prefill;

		for (const l of live) {
			if (l.phase === 'idle') continue;

			const b = bucketIndex(l.ctx, buckets);

			if (b < 0) continue;

			const hi = buckets[b].hi;

			if (
				l.decodeTps !== null &&
				t > this.peaks.decodeResetAt &&
				l.decodeTps > (decode[hi]?.v ?? 0)
			) {
				decode = { ...decode, [hi]: { t, v: l.decodeTps } };
			}

			if (l.decodeTps !== null && t > this.peaks.decodeResetAt && l.task !== undefined) {
				const key = String(l.task);
				const p = this.pendingNoCopy.get(key) ?? { at: t, best: {} };

				if (l.decodeTps > (p.best[hi]?.v ?? 0)) p.best = { ...p.best, [hi]: { t, v: l.decodeTps } };

				p.at = t;
				this.pendingNoCopy.set(key, p);
			}

			if (
				l.prefillTps !== null &&
				t > this.peaks.prefillResetAt &&
				l.prefillTps > (prefill[hi]?.v ?? 0)
			) {
				prefill = { ...prefill, [hi]: { t, v: l.prefillTps } };
			}
		}

		// requests whose record never came (a server without /requests, a restart) are let go
		for (const [key, p] of this.pendingNoCopy) {
			if (t - p.at > PENDING_NO_COPY_MS) this.pendingNoCopy.delete(key);
		}

		if (decode === this.peaks.decode && prefill === this.peaks.prefill) return;

		this.peaks = { ...this.peaks, decode, prefill };
		this.persist('peaks', this.peaks);
		this.observeRecords();
	}

	/** Every speed figure the page glows on a new best, with its key and direction. */
	private recordFigures(): [
		key: string,
		value: number | null | undefined,
		lowerIsBetter: boolean
	][] {
		const newest = this.benchView[0]?.values ?? {};
		const out: [string, number | null | undefined, boolean][] = [
			['card:prefill', this.card.prefill, false],
			['card:decode', this.card.decode, false],
			['card:copy', this.card.copy, false],
			...BENCH_CHART_KEYS.map((k): [string, number | undefined, boolean] => [
				`chart:${k}`,
				newest[k],
				k === 'session'
			]),
			['cls:prompt', this.classifier.promptPeak, false],
			['cls:fastest', this.classifier.fastest, true]
		];

		for (const b of this.buckets) {
			out.push([`decode:${b.hi}`, this.peaks.decode[b.hi]?.v ?? null, false]);
			out.push([`decode-nc:${b.hi}`, this.peaks.decodeNoCopy[b.hi]?.v ?? null, false]);
			out.push([`prefill:${b.hi}`, this.peaks.prefill[b.hi]?.v ?? null, false]);
		}

		return out;
	}

	/** What the page shows now is no record (load, model switch, first answer with history). */
	private seedRecords(): void {
		for (const [key, value] of this.recordFigures()) this.glow.seed(key, value);
	}

	private seen(status: number): boolean {
		if (status === 401 || status === 403) {
			this.auth = 'needed';

			return false;
		}

		if (status === 0) {
			this.online = false;

			return false;
		}

		this.online = true;

		if (status >= 200 && status < 300) this.auth = 'ok';

		return status >= 200 && status < 300;
	}

	// ---------------------------------------------------------------- Hugging Face

	private setProps(id: string, p: RdnaProps): void {
		this.props = { ...this.props, [id]: p };

		const label = hardwareLabel(p.devices);

		if (label && label !== loadSeenHardware(id)) saveSeenHardware(id, label);

		if (id !== this.selectedId) return;

		if (label) this.hwSeen = label;

		const disk = p.prompt_cache_disk;

		if (disk && this.cache.counters && cacheCountersWentDown(disk, this.cache.counters)) {
			// the server restarted: its counters start from zero again, and so does the baseline
			this.cache = { ...this.cache, counters: {} };
			this.persist('cache', this.cache);
		} else if (this.cache.counters === null && disk && this.cache.at > 0) {
			// after a reset made before the counters were known: the first sight is the baseline
			this.cache = { ...this.cache, counters: disk };
			this.persist('cache', this.cache);
		}

		this.lookupHf();
	}

	/** Mark a listed model as still starting (its endpoints answer with a server error) or not. */
	private setStarting(id: string, on: boolean): void {
		if (!!this.starting[id] === on) return;

		const next = { ...this.starting };

		if (on) next[id] = true;
		else delete next[id];

		this.starting = next;
	}
}

/** Re-read the runs after the bench saved one (call from an effect). */
export function benchSavedAt(): number {
	return benchRunner.savedAt;
}
