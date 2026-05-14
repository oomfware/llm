import type {
	CacheControlEphemeral,
	ContentBlockParam,
	Message,
	MessageCreateParamsNonStreaming,
	MessageCreateParamsStreaming,
	MessageParam,
	RawMessageStreamEvent,
	TextBlockParam,
	Tool,
} from '@anthropic-ai/sdk/resources/messages';

import {
	createChatAdapter,
	type ChatAdapter,
	type ChatStreamOptions,
	type StructuredOutputOptions,
	type StructuredOutputResult,
	type WireTool,
} from '../adapter.ts';
import { parseJsonSseStream, postSse } from '../internal/http.ts';
import type {
	AdapterChunk,
	AssistantContent,
	FinishReason,
	ModelMessage,
	ProviderMetadata,
	SystemContent,
	ToolContent,
	UserContent,
	Usage,
} from '../types.ts';

import type { AnthropicKnownModelId } from './generated/anthropic-models.ts';

// #region per-model type machinery

/**
 * known anthropic chat models. the `(string & {})` tail keeps autocomplete working for known entries while
 * still accepting any other model string. the literal union is generated — see `scripts/update-models.ts`.
 */
export type AnthropicModel = AnthropicKnownModelId | (string & {});

/** anthropic prompt-cache breakpoint marker. */
export interface AnthropicCacheControl {
	type: 'ephemeral';
	ttl?: '5m' | '1h';
}

/**
 * anthropic-specific provider metadata recognised by this adapter on messages and parts. extra keys are
 * passed through verbatim — only the documented ones below are interpreted.
 */
export interface AnthropicProviderMetadata {
	/** stamp `cache_control` on the wire block this metadata is attached to. */
	cacheControl?: AnthropicCacheControl;
	/** signature for an extended-thinking block — round-tripped verbatim on resume. */
	signature?: string;
	/** opaque data for a `redacted_thinking` block — round-tripped verbatim on resume. */
	data?: string;
}

interface AnthropicBaseProviderOptions {
	topK?: number;
	stopSequences?: string[];
	thinking?: { type: 'enabled'; budgetTokens: number } | { type: 'disabled' };
}

interface AnthropicChatModelProviderOptionsByName {
	'claude-opus-4-5': AnthropicBaseProviderOptions;
	'claude-sonnet-4-5': AnthropicBaseProviderOptions;
	'claude-haiku-4-5': AnthropicBaseProviderOptions;
}

type ResolveProviderOptions<TModel extends string> =
	TModel extends keyof AnthropicChatModelProviderOptionsByName
		? AnthropicChatModelProviderOptionsByName[TModel]
		: AnthropicBaseProviderOptions;

// #endregion

export interface AnthropicConfig {
	apiKey?: string;
	baseUrl?: string;
	headers?: Record<string, string>;
	fetch?: typeof fetch;
	/** anthropic-version header. defaults to '2023-06-01'. */
	version?: string;
}

export type AnthropicAdapter<TModel extends AnthropicModel = AnthropicModel> = ChatAdapter<
	TModel,
	ResolveProviderOptions<TModel>
>;

export const anthropic = <const TModel extends AnthropicModel>(
	model: TModel,
	config: AnthropicConfig = {},
): AnthropicAdapter<TModel> => {
	const apiKey = config.apiKey;
	const baseUrl = config.baseUrl ?? 'https://api.anthropic.com/v1';
	const fetcher = config.fetch ?? fetch;
	const version = config.version ?? '2023-06-01';

	return createChatAdapter<AnthropicAdapter<TModel>>({
		kind: 'chat',
		provider: 'anthropic',
		model,
		chatStream(opts) {
			return streamAnthropic({
				...opts,
				model,
				apiKey,
				baseUrl,
				headers: config.headers,
				fetcher,
				version,
			});
		},
		structuredOutput(opts) {
			return anthropicStructuredOutput({
				...opts,
				model,
				apiKey,
				baseUrl,
				headers: config.headers,
				fetcher,
				version,
			});
		},
	});
};

// #region streaming impl

interface StreamAnthropicArgs extends ChatStreamOptions<AnthropicBaseProviderOptions> {
	model: string;
	apiKey: string | undefined;
	baseUrl: string;
	headers: Record<string, string> | undefined;
	fetcher: typeof fetch;
	version: string;
}

type BlockState =
	| { kind: 'text'; id: string }
	| { kind: 'thinking'; id: string; signature: string }
	| { kind: 'redacted_thinking' }
	| { kind: 'tool_use'; id: string; toolName: string };

async function* streamAnthropic(args: StreamAnthropicArgs): AsyncGenerator<AdapterChunk, void, void> {
	const result = await postSse({
		url: `${args.baseUrl}/messages`,
		headers: {
			'content-type': 'application/json',
			...(args.apiKey ? { 'x-api-key': args.apiKey } : {}),
			'anthropic-version': args.version,
			...args.headers,
		},
		body: JSON.stringify(buildRequest(args)),
		signal: args.signal,
		fetch: args.fetcher,
		errorPrefix: 'anthropic',
	});
	if (!result.ok) {
		yield result.error;
		return;
	}

	// anthropic keys content blocks by `index` and starts/stops them around their deltas.
	const blocks = new Map<number, BlockState>();
	let finishReason: FinishReason = 'stop';
	let usage: Usage | undefined;

	for await (const event of parseJsonSseStream<RawMessageStreamEvent>(result.body)) {
		switch (event.type) {
			case 'message_start': {
				const u = event.message?.usage;
				if (u) {
					usage = toUsage(u);
				}
				break;
			}
			case 'content_block_start': {
				const cb = event.content_block;
				const id = String(event.index);
				switch (cb.type) {
					case 'text': {
						blocks.set(event.index, { kind: 'text', id });
						yield { type: 'text-start', id };
						break;
					}
					case 'thinking': {
						blocks.set(event.index, { kind: 'thinking', id, signature: '' });
						yield { type: 'reasoning-start', id };
						break;
					}
					case 'redacted_thinking': {
						blocks.set(event.index, { kind: 'redacted_thinking' });
						yield {
							type: 'redacted-reasoning',
							providerMetadata: { anthropic: { data: cb.data } satisfies AnthropicProviderMetadata },
						};
						break;
					}
					case 'tool_use': {
						blocks.set(event.index, { kind: 'tool_use', id: cb.id, toolName: cb.name });
						yield { type: 'tool-call-start', id: cb.id, name: cb.name };
						break;
					}
				}
				break;
			}
			case 'content_block_delta': {
				const block = blocks.get(event.index);
				if (!block) {
					break;
				}
				const delta = event.delta;
				switch (block.kind) {
					case 'text': {
						if (delta.type === 'text_delta' && delta.text) {
							yield { type: 'text-delta', id: block.id, delta: delta.text };
						}
						break;
					}
					case 'thinking': {
						if (delta.type === 'thinking_delta' && delta.thinking) {
							yield { type: 'reasoning-delta', id: block.id, delta: delta.thinking };
						} else if (delta.type === 'signature_delta' && delta.signature) {
							block.signature += delta.signature;
						}
						break;
					}
					case 'tool_use': {
						if (delta.type === 'input_json_delta') {
							yield {
								type: 'tool-call-delta',
								id: block.id,
								name: block.toolName,
								argsDelta: delta.partial_json,
							};
						}
						break;
					}
				}
				break;
			}
			case 'content_block_stop': {
				const block = blocks.get(event.index);
				if (!block) {
					break;
				}
				switch (block.kind) {
					case 'text': {
						yield { type: 'text-end', id: block.id };
						break;
					}
					case 'thinking': {
						const meta: AnthropicProviderMetadata = { signature: block.signature };
						yield { type: 'reasoning-end', id: block.id, providerMetadata: { anthropic: meta } };
						break;
					}
					case 'tool_use': {
						yield { type: 'tool-call-end', id: block.id, name: block.toolName };
						break;
					}
					// redacted_thinking emits the part on start, nothing to do on stop
				}
				break;
			}
			case 'message_delta': {
				if (event.delta?.stop_reason) {
					finishReason = mapFinishReason(event.delta.stop_reason);
				}
				if (event.usage) {
					usage = mergeUsage(usage, {
						inputTokens: usage?.inputTokens ?? 0,
						outputTokens: event.usage.output_tokens,
						cacheCreationInputTokens: event.usage.cache_creation_input_tokens ?? undefined,
						cacheReadInputTokens: event.usage.cache_read_input_tokens ?? undefined,
					});
				}
				break;
			}
		}
	}

	yield { type: 'finish', reason: finishReason, usage };
}

const toUsage = (u: {
	input_tokens: number;
	output_tokens: number;
	cache_creation_input_tokens?: number | null;
	cache_read_input_tokens?: number | null;
}): Usage => ({
	inputTokens: u.input_tokens,
	outputTokens: u.output_tokens,
	cacheCreationInputTokens: u.cache_creation_input_tokens ?? undefined,
	cacheReadInputTokens: u.cache_read_input_tokens ?? undefined,
});

const mergeUsage = (a: Usage | undefined, b: Usage): Usage => {
	if (!a) {
		return b;
	}
	return {
		inputTokens: b.inputTokens ?? a.inputTokens,
		outputTokens: b.outputTokens ?? a.outputTokens,
		cacheCreationInputTokens: b.cacheCreationInputTokens ?? a.cacheCreationInputTokens,
		cacheReadInputTokens: b.cacheReadInputTokens ?? a.cacheReadInputTokens,
	};
};

interface PreparedRequest {
	system: TextBlockParam[] | undefined;
	conversation: MessageParam[];
}

const prepareMessages = (messages: ModelMessage[]): PreparedRequest => {
	const systemBlocks: TextBlockParam[] = [];
	const conversation: MessageParam[] = [];

	// identify the last assistant message — for prefill, anthropic forbids
	// trailing whitespace on the final text block of the last assistant turn.
	let lastAssistantIndex = -1;
	for (let i = messages.length - 1; i >= 0; i--) {
		const m = messages[i];
		if (m && m.role === 'assistant') {
			lastAssistantIndex = i;
			break;
		}
	}
	const isPrefill = lastAssistantIndex === messages.length - 1;

	for (let i = 0; i < messages.length; i++) {
		const m = messages[i];
		if (!m) {
			continue;
		}
		switch (m.role) {
			case 'system': {
				appendSystem(systemBlocks, m.content, m.providerMetadata);
				break;
			}
			case 'user': {
				conversation.push(toUserMessage(m.content, m.providerMetadata));
				break;
			}
			case 'assistant': {
				const trimLastText = isPrefill && i === lastAssistantIndex;
				conversation.push(toAssistantMessage(m.content, m.providerMetadata, trimLastText));
				break;
			}
			case 'tool': {
				conversation.push(toToolMessage(m.content, m.providerMetadata));
				break;
			}
		}
	}

	return {
		system: systemBlocks.length > 0 ? systemBlocks : undefined,
		conversation,
	};
};

/**
 * resolve the cache_control marker for a given content part. part-level metadata wins; otherwise the
 * message-level marker applies only to the last part of the message (the cache breakpoint).
 */
const resolveCacheControl = (
	partMeta: ProviderMetadata | undefined,
	messageCacheControl: CacheControlEphemeral | undefined,
	isLast: boolean,
): CacheControlEphemeral | undefined => {
	const partCacheControl = readCacheControl(partMeta);
	return partCacheControl ?? (isLast ? messageCacheControl : undefined);
};

const appendSystem = (
	out: TextBlockParam[],
	content: SystemContent,
	messageMeta: ProviderMetadata | undefined,
): void => {
	const messageCacheControl = readCacheControl(messageMeta);
	for (let i = 0; i < content.length; i++) {
		const part = content[i];
		if (!part) {
			continue;
		}
		const block: TextBlockParam = { type: 'text', text: part.text };
		const cacheControl = resolveCacheControl(
			part.providerMetadata,
			messageCacheControl,
			i === content.length - 1,
		);
		if (cacheControl) {
			block.cache_control = cacheControl;
		}
		out.push(block);
	}
};

const toUserMessage = (content: UserContent, messageMeta: ProviderMetadata | undefined): MessageParam => {
	const blocks: ContentBlockParam[] = [];
	const messageCacheControl = readCacheControl(messageMeta);
	for (let i = 0; i < content.length; i++) {
		const part = content[i];
		if (!part) {
			continue;
		}
		const block: TextBlockParam = { type: 'text', text: part.text };
		const cacheControl = resolveCacheControl(
			part.providerMetadata,
			messageCacheControl,
			i === content.length - 1,
		);
		if (cacheControl) {
			block.cache_control = cacheControl;
		}
		blocks.push(block);
	}
	return { role: 'user', content: blocks };
};

const toAssistantMessage = (
	content: AssistantContent,
	messageMeta: ProviderMetadata | undefined,
	trimLastText: boolean,
): MessageParam => {
	const blocks: ContentBlockParam[] = [];
	const messageCacheControl = readCacheControl(messageMeta);

	// pre-compute the index of the last text block for prefill trimming
	let lastTextIndex = -1;
	if (trimLastText) {
		for (let i = content.length - 1; i >= 0; i--) {
			if (content[i]?.type === 'text') {
				lastTextIndex = i;
				break;
			}
		}
	}

	for (let i = 0; i < content.length; i++) {
		const part = content[i];
		if (!part) {
			continue;
		}
		const isLast = i === content.length - 1;
		// thinking and redacted_thinking blocks cannot carry cache_control directly
		switch (part.type) {
			case 'reasoning': {
				const meta = readAnthropicMeta(part.providerMetadata);
				const signature = meta?.signature;
				if (!signature) {
					// anthropic requires a signature on every thinking block for
					// resume; without one we'd fail validation. drop the block —
					// the caller is sending reasoning that wasn't produced by
					// anthropic, so it has no place in this turn.
					break;
				}
				blocks.push({ type: 'thinking', thinking: part.text, signature });
				break;
			}
			case 'redacted-reasoning': {
				const meta = readAnthropicMeta(part.providerMetadata);
				const data = meta?.data;
				if (!data) {
					break;
				}
				blocks.push({ type: 'redacted_thinking', data });
				break;
			}
			case 'text': {
				const text = i === lastTextIndex ? part.text.replace(/\s+$/, '') : part.text;
				const block: TextBlockParam = { type: 'text', text };
				const cacheControl = resolveCacheControl(part.providerMetadata, messageCacheControl, isLast);
				if (cacheControl) {
					block.cache_control = cacheControl;
				}
				blocks.push(block);
				break;
			}
			case 'tool-call': {
				let input: unknown = {};
				try {
					input = part.arguments.length > 0 ? JSON.parse(part.arguments) : {};
				} catch {
					// keep empty input on parse failure; the model gets to see the assistant turn anyway
				}
				const block: ContentBlockParam = { type: 'tool_use', id: part.id, name: part.name, input };
				const cacheControl = resolveCacheControl(part.providerMetadata, messageCacheControl, isLast);
				if (cacheControl) {
					block.cache_control = cacheControl;
				}
				blocks.push(block);
				break;
			}
		}
	}

	return { role: 'assistant', content: blocks };
};

const toToolMessage = (content: ToolContent, messageMeta: ProviderMetadata | undefined): MessageParam => {
	const blocks: ContentBlockParam[] = [];
	const messageCacheControl = readCacheControl(messageMeta);
	for (let i = 0; i < content.length; i++) {
		const part = content[i];
		if (!part) {
			continue;
		}
		const block: ContentBlockParam = {
			type: 'tool_result',
			tool_use_id: part.toolCallId,
			content: part.output,
		};
		if (part.isError) {
			block.is_error = true;
		}
		const cacheControl = resolveCacheControl(
			part.providerMetadata,
			messageCacheControl,
			i === content.length - 1,
		);
		if (cacheControl) {
			block.cache_control = cacheControl;
		}
		blocks.push(block);
	}
	return { role: 'user', content: blocks };
};

const readAnthropicMeta = (meta: ProviderMetadata | undefined): AnthropicProviderMetadata | undefined => {
	if (!meta) {
		return undefined;
	}
	const value = meta.anthropic;
	if (!value || typeof value !== 'object') {
		return undefined;
	}
	// oxlint-disable-next-line typescript/no-unsafe-type-assertion
	return value as AnthropicProviderMetadata;
};

const readCacheControl = (meta: ProviderMetadata | undefined): CacheControlEphemeral | undefined => {
	const anthropicMeta = readAnthropicMeta(meta);
	const cc = anthropicMeta?.cacheControl;
	if (!cc) {
		return undefined;
	}
	const out: CacheControlEphemeral = { type: 'ephemeral' };
	if (cc.ttl) {
		out.ttl = cc.ttl;
	}
	return out;
};

const buildRequest = (args: StreamAnthropicArgs): MessageCreateParamsStreaming => {
	const { system, conversation } = prepareMessages(args.messages);
	const opts = args.providerOptions ?? {};

	const body: MessageCreateParamsStreaming = {
		model: args.model,
		messages: conversation,
		max_tokens: args.maxTokens ?? 4096,
		stream: true,
		system,
		temperature: args.temperature,
		top_p: args.topP,
		top_k: opts.topK,
		stop_sequences: opts.stopSequences,
		tools: args.tools && args.tools.length > 0 ? args.tools.map(toAnthropicWireTool) : undefined,
	};

	if (opts.thinking) {
		body.thinking =
			opts.thinking.type === 'enabled'
				? { type: 'enabled', budget_tokens: opts.thinking.budgetTokens }
				: { type: 'disabled' };
	}

	return body;
};

const toAnthropicWireTool = (t: WireTool): Tool => {
	return {
		name: t.name,
		description: t.description,
		// oxlint-disable-next-line typescript/no-unsafe-type-assertion
		input_schema: t.jsonSchema as Tool.InputSchema,
	};
};

const mapFinishReason = (reason: string): FinishReason => {
	switch (reason) {
		case 'end_turn':
		case 'stop_sequence':
			return 'stop';
		case 'max_tokens':
			return 'length';
		case 'tool_use':
			return 'tool-calls';
		default:
			return 'stop';
	}
};

// #endregion

// #region structured output

interface StructuredAnthropicArgs extends StructuredOutputOptions<AnthropicBaseProviderOptions> {
	model: string;
	apiKey: string | undefined;
	baseUrl: string;
	headers: Record<string, string> | undefined;
	fetcher: typeof fetch;
	version: string;
}

const STRUCTURED_TOOL_NAME = 'submit_output';

/**
 * anthropic has no native json mode, so we coerce structured output via tool prefill: define a single tool
 * whose input schema matches the desired output, force `tool_choice` onto it, and parse the input from the
 * resulting `tool_use` block.
 */
const anthropicStructuredOutput = async (args: StructuredAnthropicArgs): Promise<StructuredOutputResult> => {
	const { system, conversation } = prepareMessages(args.messages);

	const body: MessageCreateParamsNonStreaming = {
		model: args.model,
		messages: conversation,
		max_tokens: args.maxTokens ?? 4096,
		stream: false,
		tools: [
			{
				name: STRUCTURED_TOOL_NAME,
				description: 'submit the final structured output',
				// oxlint-disable-next-line typescript/no-unsafe-type-assertion
				input_schema: args.jsonSchema as Tool.InputSchema,
			},
		],
		tool_choice: { type: 'tool', name: STRUCTURED_TOOL_NAME },
		system,
		temperature: args.temperature,
		top_p: args.topP,
	};

	const response = await args.fetcher(`${args.baseUrl}/messages`, {
		method: 'POST',
		headers: {
			'content-type': 'application/json',
			...(args.apiKey ? { 'x-api-key': args.apiKey } : {}),
			'anthropic-version': args.version,
			...args.headers,
		},
		body: JSON.stringify(body),
		signal: args.signal,
	});

	if (!response.ok) {
		const text = await response.text().catch(() => '');
		throw new Error(`anthropic: ${response.status} ${response.statusText} ${text}`);
	}

	// oxlint-disable-next-line typescript/no-unsafe-type-assertion
	const json = (await response.json()) as Message;
	const toolUse = json.content.find((c) => c.type === 'tool_use');
	if (!toolUse) {
		throw new Error('anthropic: structured output response did not include a tool_use block');
	}

	return { data: toolUse.input, rawText: JSON.stringify(toolUse.input) };
};

// #endregion
