import { BuiltInTool } from '$lib/enums';
import {
	buildWebFetchToolDefinition,
	buildWebSearchToolDefinition,
	executeWebTool,
	formatPage,
	formatSearchResults,
	isWebTool,
	WEB_FETCH_MAX_CHARS,
	WEB_SEARCH_MAX_RESULTS,
	WEB_SNIPPET_MAX_CHARS
} from '$lib/rdna/web-tools';
import { describe, expect, it, vi } from 'vitest';

function jsonResponse(body: unknown, status = 200): Response {
	return new Response(JSON.stringify(body), {
		headers: { 'Content-Type': 'application/json' },
		status
	});
}

describe('web tools', () => {
	it('defines web_search and web_fetch as browser tools with their required arguments', () => {
		const search = buildWebSearchToolDefinition();
		const fetchDef = buildWebFetchToolDefinition();

		expect(search.function.name).toBe('web_search');
		expect(fetchDef.function.name).toBe('web_fetch');
		expect(search.function.parameters.required).toEqual(['query']);
		expect(fetchDef.function.parameters.required).toEqual(['url']);
		expect(isWebTool(BuiltInTool.BROWSER_WEB_SEARCH)).toBe(true);
		expect(isWebTool(BuiltInTool.BROWSER_WEB_FETCH)).toBe(true);
		expect(isWebTool('get_datetime')).toBe(false);
	});

	it('numbers search results and cuts long snippets', () => {
		const text = formatSearchResults('q', [
			{ snippet: 'x'.repeat(1000), title: 'One', url: 'https://a.example' },
			{ description: 'about two', title: '', url: 'https://b.example' }
		]);

		expect(text).toContain('1. One\n   https://a.example\n   ');
		expect(text).toContain(`${'x'.repeat(WEB_SNIPPET_MAX_CHARS - 1)}…`);
		expect(text).not.toContain('x'.repeat(WEB_SNIPPET_MAX_CHARS));
		expect(text).toContain('2. https://b.example\n   https://b.example\n   about two');
		expect(formatSearchResults('nothing', [])).toBe('No results for "nothing".');
	});

	it('cuts a long page and says how much is left', () => {
		const page = formatPage(
			{ content: 'y'.repeat(WEB_FETCH_MAX_CHARS + 500), title: 'Page', url: 'https://p.example/' },
			'https://p.example'
		);

		expect(page.startsWith('Page\nhttps://p.example/\n\n')).toBe(true);
		expect(page).toContain('[… truncated, 500 more characters]');
		expect(formatPage({ content: '' }, 'https://e.example')).toBe(
			'https://e.example\n\n(the page has no readable text)'
		);
	});

	it('searches the keyless endpoint with the app header and a capped result count', async () => {
		const fetchImpl = vi.fn(async () =>
			jsonResponse({ results: [{ snippet: 'hit', title: 'T', url: 'https://t.example' }] })
		);
		const res = await executeWebTool(
			'web_search',
			{ max_results: 50, query: ' llama.cpp ' },
			undefined,
			fetchImpl as unknown as typeof fetch
		);
		const [url, init] = fetchImpl.mock.calls[0] as unknown as [string, RequestInit];

		expect(url).toBe('https://api.keenable.ai/v1/search/public');
		expect(init.method).toBe('POST');
		expect(JSON.parse(init.body as string)).toEqual({
			max_results: WEB_SEARCH_MAX_RESULTS,
			query: 'llama.cpp'
		});
		expect((init.headers as Record<string, string>)['X-Keenable-Title']).toBe('rdna-fast');
		expect(res).toEqual({ content: '1. T\n   https://t.example\n   hit', isError: false });
	});

	it('reads a page through the keyless fetch endpoint', async () => {
		const fetchImpl = vi.fn(async () =>
			jsonResponse({ content: 'Hello', title: 'Ex', url: 'https://example.com/' })
		);
		const res = await executeWebTool(
			'web_fetch',
			{ url: 'https://example.com' },
			undefined,
			fetchImpl as unknown as typeof fetch
		);
		const [url] = fetchImpl.mock.calls[0] as unknown as [string];

		expect(url).toBe('https://api.keenable.ai/v1/fetch/public?url=https%3A%2F%2Fexample.com');
		expect(res).toEqual({ content: 'Ex\nhttps://example.com/\n\nHello', isError: false });
	});

	it('returns readable errors for bad arguments and HTTP failures', async () => {
		const never = vi.fn();

		expect(
			await executeWebTool('web_search', {}, undefined, never as unknown as typeof fetch)
		).toEqual({ content: 'Web search needs a "query".', isError: true });
		expect(
			await executeWebTool(
				'web_fetch',
				{ url: 'file:///etc' },
				undefined,
				never as unknown as typeof fetch
			)
		).toEqual({ content: 'Web fetch needs an http(s) "url".', isError: true });
		expect(never).not.toHaveBeenCalled();

		const limited = vi.fn(async () => new Response('slow down', { status: 429 }));

		expect(
			await executeWebTool(
				'web_search',
				{ query: 'q' },
				undefined,
				limited as unknown as typeof fetch
			)
		).toEqual({ content: 'Web search failed: HTTP 429 slow down', isError: true });
	});

	it('throws when the user stops the chat, as every tool does', async () => {
		const ctrl = new AbortController();
		const fetchImpl = vi.fn(async (_url: string, init?: RequestInit) => {
			ctrl.abort();
			init?.signal?.throwIfAborted();

			return jsonResponse({});
		});

		await expect(
			executeWebTool(
				'web_search',
				{ query: 'q' },
				ctrl.signal,
				fetchImpl as unknown as typeof fetch
			)
		).rejects.toThrow();
	});
});
