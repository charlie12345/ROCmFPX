/**
 * RDNA-FAST status page: Hugging Face lookups for the model header and the speed card, cached in
 * localStorage. A repo guessed from the file path is linked only when the Hub confirms it; the
 * "with <base model>" line and the avatar come from the end of its `cardData.base_model` chain
 * (ukisai/Swift-1.5-Qwen3.8-27B-GGUF -> ukisai/Swift-1.5-Qwen3.8-27b -> Qwen/Qwen3.8-27B).
 * Avatars are kept as data URLs so the exported PNG never needs a cross-origin fetch.
 * Nothing is looked up while the UI's "Use the Hugging Face Hub API" setting is off.
 */

import { STORAGE_PREFIX } from './constants';
import { readJson, writeJson } from './storage';
import { HF_API_MODELS_URL, HF_AVATARS_URL } from '$lib/constants/huggingface.constants';
import { HuggingFaceService } from '$lib/services/huggingface.service';

const CACHE_KEY = `${STORAGE_PREFIX}:hf`;
const FOUND_TTL_MS = 7 * 24 * 3600 * 1000;
const MISSING_TTL_MS = 24 * 3600 * 1000;
const FAILED_TTL_MS = 10 * 60 * 1000;
const MAX_CHAIN = 6;
const MAX_AVATAR_BYTES = 200 * 1024;

export interface HfRepoInfo {
	exists: boolean;
	/** The last repo of the base model chain (the repo itself when it names none). */
	root: string | null;
	t: number;
	failed?: boolean;
}

interface HfCache {
	repos: Record<string, HfRepoInfo>;
	avatars: Record<string, { data: string | null; t: number }>;
}

function loadCache(): HfCache {
	const v = readJson<HfCache>(CACHE_KEY);

	return {
		avatars: v?.avatars && typeof v.avatars === 'object' ? v.avatars : {},
		repos: v?.repos && typeof v.repos === 'object' ? v.repos : {}
	};
}

function saveCache(c: HfCache): void {
	writeJson(CACHE_KEY, c);
}

function fresh(t: number, ttl: number): boolean {
	return Date.now() - t < ttl;
}

export function hfEnabled(): boolean {
	try {
		return HuggingFaceService.isEnabled();
	} catch {
		return false;
	}
}

/** The first `cardData.base_model` of a model card (a string or a list). */
export function firstBaseModel(details: unknown): string | null {
	if (!details || typeof details !== 'object') return null;

	const card = (details as { cardData?: { base_model?: unknown } }).cardData;
	const base = card?.base_model;

	if (typeof base === 'string' && base.trim()) return base.trim();

	if (Array.isArray(base) && typeof base[0] === 'string' && base[0].trim()) return base[0].trim();

	return null;
}

/** Org and name of a repo id. */
export function splitRepo(repo: string): { org: string; name: string } {
	const i = repo.indexOf('/');

	return i < 0 ? { name: repo, org: '' } : { name: repo.slice(i + 1), org: repo.slice(0, i) };
}

async function fetchDetails(repo: string): Promise<{ status: number; data: unknown }> {
	const res = await fetch(`${HF_API_MODELS_URL}/${repo}`);

	if (!res.ok) return { data: null, status: res.status };

	const payload = (await res.json()) as unknown;

	return { data: Array.isArray(payload) ? payload[0] : payload, status: res.status };
}

const pendingRepos = new Map<string, Promise<HfRepoInfo | null>>();

/** Look a repo up (exists, base chain root), from the cache when fresh. */
export function lookupRepo(repo: string): Promise<HfRepoInfo | null> {
	if (!repo || !hfEnabled()) return Promise.resolve(null);

	const cached = loadCache().repos[repo];

	if (cached) {
		const ttl = cached.failed ? FAILED_TTL_MS : cached.exists ? FOUND_TTL_MS : MISSING_TTL_MS;

		if (fresh(cached.t, ttl)) return Promise.resolve(cached);
	}

	const pending = pendingRepos.get(repo);

	if (pending) return pending;

	const p = (async (): Promise<HfRepoInfo | null> => {
		let info: HfRepoInfo;

		try {
			const first = await fetchDetails(repo);

			if (first.status === 404 || first.status === 401) {
				info = { exists: false, root: null, t: Date.now() };
			} else if (first.status !== 200) {
				info = { exists: false, failed: true, root: null, t: Date.now() };
			} else {
				let root = repo;
				let base = firstBaseModel(first.data);

				const seen = new Set([repo.toLowerCase()]);

				for (let i = 0; base && i < MAX_CHAIN && !seen.has(base.toLowerCase()); i++) {
					seen.add(base.toLowerCase());
					root = base;

					const next = await fetchDetails(base);

					if (next.status !== 200) break;

					base = firstBaseModel(next.data);
				}

				info = { exists: true, root, t: Date.now() };
			}
		} catch {
			info = { exists: false, failed: true, root: null, t: Date.now() };
		}

		const c = loadCache();

		c.repos[repo] = info;
		saveCache(c);

		return info;
	})().finally(() => pendingRepos.delete(repo));

	pendingRepos.set(repo, p);

	return p;
}

function blobToDataUrl(blob: Blob): Promise<string> {
	return new Promise((resolve, reject) => {
		const reader = new FileReader();

		reader.onload = () => resolve(String(reader.result));
		reader.onerror = () => reject(reader.error);
		reader.readAsDataURL(blob);
	});
}

const pendingAvatars = new Map<string, Promise<string | null>>();

/** The avatar of an org or user as a data URL, null when there is none. */
export function lookupAvatar(org: string): Promise<string | null> {
	if (!org || !hfEnabled()) return Promise.resolve(null);

	const cached = loadCache().avatars[org];

	if (cached && fresh(cached.t, cached.data ? FOUND_TTL_MS : MISSING_TTL_MS)) {
		return Promise.resolve(cached.data);
	}

	const pending = pendingAvatars.get(org);

	if (pending) return pending;

	const p = (async (): Promise<string | null> => {
		let data: string | null = null;

		try {
			const res = await fetch(`${HF_AVATARS_URL}/${org}`);

			if (res.ok) {
				const blob = await res.blob();

				if (blob.type.startsWith('image/') && blob.size <= MAX_AVATAR_BYTES) {
					data = await blobToDataUrl(blob);
				}
			}
		} catch {
			// offline: no avatar, try again after the short TTL
			return null;
		}

		const c = loadCache();

		c.avatars[org] = { data, t: Date.now() };
		saveCache(c);

		return data;
	})().finally(() => pendingAvatars.delete(org));

	pendingAvatars.set(org, p);

	return p;
}
