/**
 * RDNA-FAST status page: GETs against the server with the UI's API key. Never throws: the caller
 * gets the HTTP status (0 = unreachable) and decides (401 asks for the key, 404 = the server does
 * not have that endpoint yet, everything else shows a dash).
 */

import { base } from '$app/paths';
import { getAuthHeaders } from '$lib/utils/api-headers';

export interface ApiResult<T> {
	ok: boolean;
	status: number;
	data: T | null;
}

export function apiUrl(path: string, params?: Record<string, string | undefined>): string {
	const q = new URLSearchParams();

	for (const [k, v] of Object.entries(params ?? {})) {
		if (v !== undefined) q.set(k, v);
	}

	const qs = q.toString();

	return `${base}${path}${qs ? `?${qs}` : ''}`;
}

export async function rdnaGet<T>(
	path: string,
	params?: Record<string, string | undefined>,
	signal?: AbortSignal
): Promise<ApiResult<T>> {
	try {
		const res = await fetch(apiUrl(path, params), {
			cache: 'no-store',
			headers: getAuthHeaders(),
			signal
		});

		if (!res.ok) return { data: null, ok: false, status: res.status };

		return { data: (await res.json()) as T, ok: true, status: res.status };
	} catch {
		return { data: null, ok: false, status: 0 };
	}
}

/**
 * Query for a per-model endpoint: `model` in router mode, and `autoload=false` so a poll never
 * loads a model the router has unloaded.
 */
export function modelQuery(
	modelId: string | null,
	routerMode: boolean
): Record<string, string | undefined> {
	return routerMode && modelId ? { autoload: 'false', model: modelId } : {};
}
