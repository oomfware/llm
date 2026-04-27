import type { StreamChunk } from '../src/types.ts';

/**
 * build a fake `Response` whose body streams the given lines as SSE
 * `data: <line>\n\n` events. each entry is one event payload (typically a
 * `JSON.stringify(...)` or the literal `'[DONE]'`).
 */
export const sseResponse = (events: string[]): Response => {
	const enc = new TextEncoder();
	const body = new ReadableStream<Uint8Array>({
		start(controller) {
			for (const e of events) {
				controller.enqueue(enc.encode(`data: ${e}\n\n`));
			}
			controller.close();
		},
	});
	return new Response(body, {
		status: 200,
		headers: { 'content-type': 'text/event-stream' },
	});
};

/** drain an async iterable into an array. */
export const collect = async <T extends StreamChunk<any>>(stream: AsyncIterable<T>): Promise<T[]> => {
	const out: T[] = [];
	for await (const c of stream) {
		out.push(c);
	}
	return out;
};

// #region anthropic event builders

interface AnthropicTextOptions {
	id?: string;
	text: string;
	usage?: { input_tokens?: number; output_tokens?: number };
	stopReason?: string;
}

/**
 * scripted SSE events for a single anthropic text content block, optionally
 * with `message_start`/`message_delta` framing.
 */
export const anthropicTextEvents = (opts: AnthropicTextOptions): string[] => {
	const events: string[] = [];
	if (opts.usage?.input_tokens !== undefined) {
		events.push(
			JSON.stringify({
				type: 'message_start',
				message: {
					id: opts.id ?? 'msg_1',
					type: 'message',
					role: 'assistant',
					content: [],
					model: 'claude',
					usage: { input_tokens: opts.usage.input_tokens, output_tokens: 0 },
				},
			}),
		);
	}
	events.push(
		JSON.stringify({ type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
		JSON.stringify({
			type: 'content_block_delta',
			index: 0,
			delta: { type: 'text_delta', text: opts.text },
		}),
		JSON.stringify({ type: 'content_block_stop', index: 0 }),
	);
	if (opts.stopReason || opts.usage?.output_tokens !== undefined) {
		events.push(
			JSON.stringify({
				type: 'message_delta',
				delta: { stop_reason: opts.stopReason ?? 'end_turn' },
				usage: { output_tokens: opts.usage?.output_tokens ?? 0 },
			}),
		);
	}
	return events;
};

interface AnthropicToolUseOptions {
	index?: number;
	id: string;
	name: string;
	args: object;
}

/** SSE events for a single anthropic `tool_use` content block. */
export const anthropicToolUseEvents = (opts: AnthropicToolUseOptions): string[] => {
	const idx = opts.index ?? 0;
	return [
		JSON.stringify({
			type: 'content_block_start',
			index: idx,
			content_block: { type: 'tool_use', id: opts.id, name: opts.name, input: {} },
		}),
		JSON.stringify({
			type: 'content_block_delta',
			index: idx,
			delta: { type: 'input_json_delta', partial_json: JSON.stringify(opts.args) },
		}),
		JSON.stringify({ type: 'content_block_stop', index: idx }),
	];
};

interface AnthropicThinkingOptions {
	index?: number;
	text: string;
	signature: string;
}

/** SSE events for a single anthropic `thinking` content block. */
export const anthropicThinkingEvents = (opts: AnthropicThinkingOptions): string[] => {
	const idx = opts.index ?? 0;
	return [
		JSON.stringify({
			type: 'content_block_start',
			index: idx,
			content_block: { type: 'thinking', thinking: '' },
		}),
		JSON.stringify({
			type: 'content_block_delta',
			index: idx,
			delta: { type: 'thinking_delta', thinking: opts.text },
		}),
		JSON.stringify({
			type: 'content_block_delta',
			index: idx,
			delta: { type: 'signature_delta', signature: opts.signature },
		}),
		JSON.stringify({ type: 'content_block_stop', index: idx }),
	];
};

// #endregion

// #region openai event builders

interface OpenAITextOptions {
	id?: string;
	text: string;
	usage?: { input_tokens?: number; output_tokens?: number };
}

/** SSE events for a single openai responses-api text message item. */
export const openaiTextEvents = (opts: OpenAITextOptions): string[] => {
	const id = opts.id ?? 'msg_1';
	return [
		JSON.stringify({
			type: 'response.output_item.added',
			item: { type: 'message', id, role: 'assistant', content: [] },
		}),
		JSON.stringify({
			type: 'response.output_text.delta',
			item_id: id,
			delta: opts.text,
		}),
		JSON.stringify({
			type: 'response.output_item.done',
			item: {
				type: 'message',
				id,
				role: 'assistant',
				status: 'completed',
				content: [{ type: 'output_text', text: opts.text, annotations: [] }],
			},
		}),
		JSON.stringify({
			type: 'response.completed',
			response: {
				status: 'completed',
				usage: {
					input_tokens: opts.usage?.input_tokens ?? 0,
					output_tokens: opts.usage?.output_tokens ?? 0,
				},
			},
		}),
		'[DONE]',
	];
};

interface OpenAIFunctionCallOptions {
	id?: string;
	callId: string;
	name: string;
	args: object;
}

/** SSE events for a single openai responses-api `function_call` item. */
export const openaiFunctionCallEvents = (opts: OpenAIFunctionCallOptions): string[] => {
	const id = opts.id ?? `fc_${opts.callId}`;
	const argsStr = JSON.stringify(opts.args);
	return [
		JSON.stringify({
			type: 'response.output_item.added',
			item: { type: 'function_call', id, call_id: opts.callId, name: opts.name, arguments: '' },
		}),
		JSON.stringify({
			type: 'response.function_call_arguments.delta',
			item_id: id,
			delta: argsStr,
		}),
		JSON.stringify({
			type: 'response.function_call_arguments.done',
			item_id: id,
			arguments: argsStr,
		}),
		JSON.stringify({
			type: 'response.output_item.done',
			item: {
				type: 'function_call',
				id,
				call_id: opts.callId,
				name: opts.name,
				arguments: argsStr,
				status: 'completed',
			},
		}),
		JSON.stringify({ type: 'response.completed', response: { status: 'completed' } }),
		'[DONE]',
	];
};

// #endregion
