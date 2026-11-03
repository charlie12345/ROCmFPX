/**
 * RDNA-FAST browser tools (owner, 2026-11-03): web_search and web_fetch on Keenable's keyless public
 * endpoints (https://keenable.ai). The browser calls them directly (Keenable allows cross-origin
 * calls), so the server needs nothing. Only the model's query or URL leaves the browser, never the
 * conversation. They sit in the chat's Browser tools next to get_datetime.
 */

import { BuiltInTool, JsonSchemaType, ToolCallType } from '$lib/enums';
import type { OpenAIToolDefinition, ToolExecutionResult } from '$lib/types';

const KEENABLE_API = 'https://api.keenable.ai';
/** Keenable asks every client to name itself. */
const KEENABLE_TITLE = 'rdna-fast';
const TIMEOUT_MS = 30_000;

export const WEB_SEARCH_DEFAULT_RESULTS = 5;
export const WEB_SEARCH_MAX_RESULTS = 10;
/** A result's snippet is cut here, so ten results stay a few thousand tokens. */
export const WEB_SNIPPET_MAX_CHARS = 400;
/** A fetched page is cut here (about 3k tokens), so one page cannot flood the context. */
export const WEB_FETCH_MAX_CHARS = 12_000;

export function buildWebSearchToolDefinition(): OpenAIToolDefinition {
	return {
		function: {
			description: 'Search the web. Returns titles, URLs and snippets.',
			name: BuiltInTool.BROWSER_WEB_SEARCH,
			parameters: {
				properties: {
					max_results: {
						description: `1-${WEB_SEARCH_MAX_RESULTS}, default ${WEB_SEARCH_DEFAULT_RESULTS}`,
						type: JsonSchemaType.NUMBER
					},
					query: { type: JsonSchemaType.STRING }
				},
				required: ['query'],
				type: JsonSchemaType.OBJECT
			}
		},
		type: ToolCallType.FUNCTION
	};
}

export function buildWebFetchToolDefinition(): OpenAIToolDefinition {
	return {
		function: {
			description: 'Read a web page as text.',
			name: BuiltInTool.BROWSER_WEB_FETCH,
			parameters: {
				properties: { url: { type: JsonSchemaType.STRING } },
				required: ['url'],
				type: JsonSchemaType.OBJECT
			}
		},
		type: ToolCallType.FUNCTION
	};
}

export function isWebTool(name: string): boolean {
	return name === BuiltInTool.BROWSER_WEB_SEARCH || name === BuiltInTool.BROWSER_WEB_FETCH;
}

interface SearchResult {
	title?: string;
	url?: string;
	snippet?: string;
	description?: string;
}

function clip(text: string, max: number): string {
	const t = text.replace(/\s+/g, ' ').trim();

	return t.length > max ? `${t.slice(0, max - 1)}…` : t;
}

/** Numbered results: title, URL, snippet. */
export function formatSearchResults(query: string, results: readonly SearchResult[]): string {
	if (results.length === 0) return `No results for "${query}".`;

	return results
		.map((r, i) => {
			const snippet = clip(r.snippet || r.description || '', WEB_SNIPPET_MAX_CHARS);

			return [
				`${i + 1}. ${r.title || r.url || 'Untitled'}`,
				`   ${r.url ?? ''}`,
				snippet ? `   ${snippet}` : ''
			]
				.filter(Boolean)
				.join('\n');
		})
		.join('\n\n');
}

/** Title, URL and the page text, cut at WEB_FETCH_MAX_CHARS with a note. */
export function formatPage(
	page: { url?: string; title?: string; content?: string },
	url: string
): string {
	const content = (page.content ?? '').trim();
	const cut = content.length > WEB_FETCH_MAX_CHARS;
	const body = cut
		? `${content.slice(0, WEB_FETCH_MAX_CHARS)}\n\n[… truncated, ${content.length - WEB_FETCH_MAX_CHARS} more characters]`
		: content || '(the page has no readable text)';
	const title = page.title?.trim();

	return [...(title ? [title] : []), page.url || url, '', body].join('\n');
}

function fail(message: string): ToolExecutionResult {
	return { content: message, isError: true };
}

function argText(args: Record<string, unknown>, key: string): string {
	const v = args[key];

	return typeof v === 'string' ? v.trim() : '';
}

/**
 * Run web_search or web_fetch. A stop by the user (the signal) is thrown as an AbortError, as for
 * every tool; anything else (bad arguments, HTTP errors, a 30 s timeout) is an error result the
 * model can read.
 */
export async function executeWebTool(
	name: string,
	args: Record<string, unknown>,
	signal?: AbortSignal,
	fetchImpl: typeof fetch = fetch
): Promise<ToolExecutionResult> {
	const search = name === BuiltInTool.BROWSER_WEB_SEARCH;
	const what = search ? 'Web search' : 'Web fetch';

	let request: { url: string; init: RequestInit };

	if (search) {
		const query = argText(args, 'query');

		if (!query) return fail('Web search needs a "query".');

		const n = Number(args.max_results);
		const maxResults = Number.isFinite(n)
			? Math.min(WEB_SEARCH_MAX_RESULTS, Math.max(1, Math.round(n)))
			: WEB_SEARCH_DEFAULT_RESULTS;

		request = {
			init: {
				body: JSON.stringify({ max_results: maxResults, query }),
				headers: { 'Content-Type': 'application/json', 'X-Keenable-Title': KEENABLE_TITLE },
				method: 'POST'
			},
			url: `${KEENABLE_API}/v1/search/public`
		};
	} else {
		const url = argText(args, 'url');

		if (!/^https?:\/\/\S+$/i.test(url)) return fail('Web fetch needs an http(s) "url".');

		request = {
			init: { headers: { 'X-Keenable-Title': KEENABLE_TITLE }, method: 'GET' },
			url: `${KEENABLE_API}/v1/fetch/public?url=${encodeURIComponent(url)}`
		};
	}

	const timeout = AbortSignal.timeout(TIMEOUT_MS);
	const linked = signal ? AbortSignal.any([signal, timeout]) : timeout;

	try {
		const res = await fetchImpl(request.url, { ...request.init, signal: linked });

		if (!res.ok) {
			const detail = (await res.text().catch(() => '')).slice(0, 200).trim();

			return fail(`${what} failed: HTTP ${res.status}${detail ? ` ${detail}` : ''}`);
		}

		const data = (await res.json()) as Record<string, unknown>;

		if (search) {
			const results = Array.isArray(data.results) ? (data.results as SearchResult[]) : [];

			return { content: formatSearchResults(argText(args, 'query'), results), isError: false };
		}

		return {
			content: formatPage(
				data as { url?: string; title?: string; content?: string },
				argText(args, 'url')
			),
			isError: false
		};
	} catch (e) {
		if (signal?.aborted) throw e;

		if (timeout.aborted) return fail(`${what} failed: no answer within ${TIMEOUT_MS / 1000} s`);

		return fail(`${what} failed: ${e instanceof Error ? e.message : String(e)}`);
	}
}
