import type { AdapterChunk } from '../types.ts';

import { parseSseStream } from './sse.ts';

export interface PostSseArgs {
	url: string;
	headers: Record<string, string>;
	body: string;
	signal: AbortSignal | undefined;
	fetch: typeof fetch;
	/** prefix for error messages, e.g. `'openai'`, `'anthropic'`. */
	errorPrefix: string;
}

export type PostSseResult =
	| { ok: true; body: ReadableStream<Uint8Array> }
	| { ok: false; error: AdapterChunk & { type: 'error' } };

/**
 * post a request that returns an SSE stream. handles network errors and non-2xx responses uniformly across
 * providers, returning either the response body for streaming or an `error` chunk to yield.
 */
export const postSse = async (args: PostSseArgs): Promise<PostSseResult> => {
	let response: Response;
	try {
		response = await args.fetch(args.url, {
			method: 'POST',
			headers: args.headers,
			body: args.body,
			signal: args.signal,
		});
	} catch (e) {
		return { ok: false, error: { type: 'error', error: e } };
	}

	if (!response.ok || !response.body) {
		const text = response.body ? await response.text().catch(() => '') : '';
		return {
			ok: false,
			error: {
				type: 'error',
				error: new Error(`${args.errorPrefix}: ${response.status} ${response.statusText} ${text}`),
			},
		};
	}

	return { ok: true, body: response.body };
};

/**
 * iterate JSON-encoded SSE events. malformed events are silently skipped.
 *
 * @param sentinel if provided, breaks the stream when this exact data payload is received (e.g. OpenAI's
 *   `[DONE]`).
 */
export async function* parseJsonSseStream<T>(
	stream: ReadableStream<Uint8Array>,
	sentinel?: string,
): AsyncGenerator<T, void, void> {
	for await (const data of parseSseStream(stream)) {
		if (sentinel !== undefined && data === sentinel) {
			break;
		}
		try {
			// oxlint-disable-next-line typescript/no-unsafe-type-assertion
			yield JSON.parse(data) as T;
		} catch {
			// malformed sse event payload; skip
		}
	}
}
