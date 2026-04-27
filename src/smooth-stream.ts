import type { AnyTool } from './tool.ts';
import type { StreamChunk } from './types.ts';

/**
 * how `text-delta` chunks are split into emitted units.
 *
 * - `'word'`: emit at whitespace boundaries (each word plus its trailing
 *   whitespace).
 * - `'line'`: emit at newline boundaries (each line including the trailing
 *   `\n`).
 * - `RegExp`: emit when the regex matches against the buffered text. the
 *   matched text plus everything before it is flushed as one chunk. the
 *   regex is normalised to non-global; supplying `/g` is fine.
 */
export type SmoothChunking = 'word' | 'line' | RegExp;

export interface SmoothStreamOptions {
	chunking?: SmoothChunking;
	/** delay in ms inserted between emitted text-delta chunks. defaults to `0`. */
	delayMs?: number;
}

/**
 * smooth a chat stream's `text-delta` chunks into larger, more ui-friendly
 * units (words, lines, or a custom regex boundary). non-text chunks
 * (`reasoning-delta`, tool chunks, `finish`, `error`) pass through unchanged
 * and *in order* — pending text is flushed before each non-text chunk so the
 * downstream consumer sees a consistent timeline.
 *
 * preserves the public {@link StreamChunk} generic so tool chunk narrowing
 * still works downstream.
 *
 * @example
 * ```ts
 * import { chat, openai, smoothStream } from '@oomfware/ai';
 *
 * for await (const chunk of smoothStream(chat({ adapter, messages }), { chunking: 'word' })) {
 *   if (chunk.type === 'text-delta') {
 *     process.stdout.write(chunk.delta);
 *   }
 * }
 * ```
 */
export const smoothStream = <TTools extends Record<string, AnyTool> = {}>(
	stream: AsyncIterable<StreamChunk<TTools>>,
	options: SmoothStreamOptions = {},
): AsyncIterable<StreamChunk<TTools>> => runSmooth(stream, options);

// oxlint-disable-next-line typescript/no-unsafe-type-assertion
const textDelta = <TTools extends Record<string, AnyTool>>(delta: string): StreamChunk<TTools> =>
	({ type: 'text-delta', delta }) as StreamChunk<TTools>;

async function* runSmooth<TTools extends Record<string, AnyTool>>(
	stream: AsyncIterable<StreamChunk<TTools>>,
	options: SmoothStreamOptions,
): AsyncGenerator<StreamChunk<TTools>, void, void> {
	const matcher = toMatcher(options.chunking ?? 'word');
	const delayMs = options.delayMs ?? 0;

	let buffer = '';

	for await (const chunk of stream) {
		if (chunk.type === 'text-delta') {
			buffer += chunk.delta;
			for (;;) {
				const match = matcher.exec(buffer);
				if (!match || match[0].length === 0) {
					break;
				}
				const cut = match.index + match[0].length;
				const out = buffer.slice(0, cut);
				buffer = buffer.slice(cut);
				yield textDelta<TTools>(out);
				if (delayMs > 0) {
					// oxlint-disable-next-line eslint/no-await-in-loop
					await sleep(delayMs);
				}
			}
			continue;
		}

		// any non-text-delta chunk: flush remaining buffered text first so the
		// downstream consumer sees the text emitted before the boundary event.
		if (buffer.length > 0) {
			yield textDelta<TTools>(buffer);
			buffer = '';
		}
		yield chunk;
	}

	if (buffer.length > 0) {
		yield textDelta<TTools>(buffer);
	}
}

const toMatcher = (chunking: SmoothChunking): RegExp => {
	switch (chunking) {
		case 'word': {
			return /\S+\s+/;
		}
		case 'line': {
			return /[^\n]*\n/;
		}
		default: {
			// strip the global flag so .exec() is stateless across iterations.
			return chunking.flags.includes('g')
				? new RegExp(chunking.source, chunking.flags.replace('g', ''))
				: chunking;
		}
	}
};

const sleep = (ms: number): Promise<void> =>
	new Promise((resolve) => {
		setTimeout(resolve, ms);
	});
