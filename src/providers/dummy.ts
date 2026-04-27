import {
	createChatAdapter,
	type ChatAdapter,
	type ChatStreamOptions,
	type StructuredOutputOptions,
	type StructuredOutputResult,
} from '../adapter.ts';
import type { AdapterChunk, FinishReason, ProviderMetadata, Usage } from '../types.ts';

export interface DummyToolCall {
	id: string;
	name: string;
	arguments: object | string;
	providerMetadata?: ProviderMetadata;
}

/**
 * a single ordered part the dummy adapter should emit. covers the same
 * variants the real adapters can produce, so tests can script reasoning +
 * text + tool-call interleavings.
 */
export type DummyPart =
	| { type: 'text'; text: string; providerMetadata?: ProviderMetadata }
	| { type: 'reasoning'; text: string; providerMetadata?: ProviderMetadata }
	| { type: 'redacted-reasoning'; providerMetadata: ProviderMetadata }
	| {
			type: 'tool-call';
			id: string;
			name: string;
			arguments: object | string;
			providerMetadata?: ProviderMetadata;
	  };

/**
 * a single scripted response returned by the dummy adapter on the next call
 * to `chatStream`. one entry per agent-loop iteration.
 *
 * if `parts` is omitted, `text` and `toolCalls` shorthand are expanded into
 * parts in that order. when a response has tool calls (in either form), the
 * default finish reason is `tool-calls`; otherwise `stop`.
 */
export interface DummyResponse {
	parts?: DummyPart[];
	text?: string;
	toolCalls?: DummyToolCall[];
	finishReason?: FinishReason;
	error?: unknown;
	usage?: Usage;
	providerMetadata?: ProviderMetadata;
}

export interface DummyConfig {
	responses: DummyResponse[];
	/**
	 * how the adapter chunks text. `char` simulates streaming token-by-token,
	 * `whole` emits the entire text in one delta. defaults to `char`.
	 */
	chunking?: 'char' | 'whole';
	/**
	 * scripted return value for `structuredOutput()` calls. consumed in order
	 * across calls — one entry per call.
	 */
	structuredOutputs?: unknown[];
}

export interface DummyAdapter extends ChatAdapter<'dummy', Record<string, unknown>> {
	/** every chatStream invocation, in order. lets tests assert on what reached the adapter. */
	readonly calls: ReadonlyArray<ChatStreamOptions<Record<string, unknown>>>;
	/** every structuredOutput invocation, in order. */
	readonly structuredCalls: ReadonlyArray<StructuredOutputOptions<Record<string, unknown>>>;
}

export const dummy = (config: DummyConfig): DummyAdapter => {
	const calls: ChatStreamOptions<Record<string, unknown>>[] = [];
	const structuredCalls: StructuredOutputOptions<Record<string, unknown>>[] = [];
	let cursor = 0;
	let structuredCursor = 0;

	return createChatAdapter<DummyAdapter>({
		kind: 'chat',
		provider: 'dummy',
		model: 'dummy',
		calls,
		structuredCalls,
		chatStream(opts) {
			// snapshot so assertions see the messages as they were at call time, not
			// after `chat()` mutates the working array on subsequent iterations.
			calls.push({ ...opts, messages: [...opts.messages], tools: opts.tools ? [...opts.tools] : undefined });
			const response = config.responses[cursor++];
			return script(response, config.chunking ?? 'char');
		},
		async structuredOutput(opts) {
			structuredCalls.push({ ...opts, messages: [...opts.messages] });
			const data = config.structuredOutputs?.[structuredCursor++];
			if (data === undefined) {
				throw new Error('dummy adapter ran out of scripted structured outputs');
			}
			const result: StructuredOutputResult = { data, rawText: JSON.stringify(data) };
			return result;
		},
	});
};

const expandResponse = (response: DummyResponse): DummyPart[] => {
	if (response.parts) {
		return response.parts;
	}
	const parts: DummyPart[] = [];
	if (response.text) {
		parts.push({ type: 'text', text: response.text });
	}
	if (response.toolCalls) {
		for (const call of response.toolCalls) {
			parts.push({
				type: 'tool-call',
				id: call.id,
				name: call.name,
				arguments: call.arguments,
				...(call.providerMetadata ? { providerMetadata: call.providerMetadata } : {}),
			});
		}
	}
	return parts;
};

const hasToolCall = (parts: DummyPart[]): boolean => parts.some((p) => p.type === 'tool-call');

async function* script(
	response: DummyResponse | undefined,
	chunking: 'char' | 'whole',
): AsyncGenerator<AdapterChunk, void, void> {
	if (!response) {
		yield { type: 'error', error: new Error('dummy adapter ran out of scripted responses') };
		return;
	}

	if (response.error !== undefined) {
		yield { type: 'error', error: response.error };
		return;
	}

	const parts = expandResponse(response);
	let textCounter = 0;
	let reasoningCounter = 0;

	for (const part of parts) {
		switch (part.type) {
			case 'text': {
				const id = `t${textCounter++}`;
				yield { type: 'text-start', id };
				if (chunking === 'whole') {
					yield { type: 'text-delta', id, delta: part.text };
				} else {
					for (const ch of part.text) {
						yield { type: 'text-delta', id, delta: ch };
					}
				}
				yield { type: 'text-end', id, providerMetadata: part.providerMetadata };
				break;
			}
			case 'reasoning': {
				const id = `r${reasoningCounter++}`;
				yield { type: 'reasoning-start', id };
				if (chunking === 'whole') {
					yield { type: 'reasoning-delta', id, delta: part.text };
				} else {
					for (const ch of part.text) {
						yield { type: 'reasoning-delta', id, delta: ch };
					}
				}
				yield { type: 'reasoning-end', id, providerMetadata: part.providerMetadata };
				break;
			}
			case 'redacted-reasoning': {
				yield { type: 'redacted-reasoning', providerMetadata: part.providerMetadata };
				break;
			}
			case 'tool-call': {
				yield { type: 'tool-call-start', id: part.id, name: part.name };
				const argsStr = typeof part.arguments === 'string' ? part.arguments : JSON.stringify(part.arguments);
				yield { type: 'tool-call-delta', id: part.id, name: part.name, argsDelta: argsStr };
				yield {
					type: 'tool-call-end',
					id: part.id,
					name: part.name,
					providerMetadata: part.providerMetadata,
				};
				break;
			}
		}
	}

	yield {
		type: 'finish',
		reason: response.finishReason ?? (hasToolCall(parts) ? 'tool-calls' : 'stop'),
		usage: response.usage,
		providerMetadata: response.providerMetadata,
	};
}
