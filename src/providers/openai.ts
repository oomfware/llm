import type {
	FunctionTool,
	Response,
	ResponseCreateParamsNonStreaming,
	ResponseCreateParamsStreaming,
	ResponseInput,
	ResponseInputItem,
	ResponseStreamEvent,
} from 'openai/resources/responses/responses';

import {
	createChatAdapter,
	type ChatAdapter,
	type ChatStreamOptions,
	type StructuredOutputOptions,
	type StructuredOutputResult,
	type WireTool,
} from '../adapter.ts';
import { parseJsonSseStream, postJson, postSse } from '../internal/http.ts';
import { makeOpenAIStrictCompatible, stripNulls } from '../internal/openai-strict.ts';
import { readProviderMeta } from '../internal/provider-metadata.ts';
import type {
	AdapterChunk,
	AssistantContent,
	FinishReason,
	ModelMessage,
	ProviderMetadata,
	Usage,
} from '../types.ts';

import type { OpenAIKnownChatModelId } from './generated/openai-models.ts';

// #region per-model type machinery

/**
 * known OpenAI chat models. the `(string & {})` tail keeps autocomplete working for known entries while still
 * accepting any other model string (e.g. fine-tunes, future releases). the literal union is generated — see
 * `scripts/update-models.ts`.
 */
export type OpenAIChatModel = OpenAIKnownChatModelId | (string & {});

/**
 * the user-facing shape mirrors the Responses API `text.format` wire shape directly so it can be passed
 * through without conversion.
 */
interface OpenAIJsonSchemaResponseFormat {
	type: 'json_schema';
	name: string;
	schema: Record<string, unknown>;
	description?: string;
	strict?: boolean | null;
}

interface OpenAIBaseProviderOptions {
	user?: string;
	parallelToolCalls?: boolean;
	responseFormat?: { type: 'json_object' } | OpenAIJsonSchemaResponseFormat;
}

interface OpenAIReasoningProviderOptions extends OpenAIBaseProviderOptions {
	reasoningEffort?: 'low' | 'medium' | 'high';
	/**
	 * request `include: ['reasoning.encrypted_content']` so reasoning items round-trip on resume without
	 * server-side state. defaults to true when `reasoningEffort` is set.
	 */
	includeEncryptedReasoning?: boolean;
}

/**
 * openai-specific provider metadata recognised by this adapter on assistant messages and parts. extra keys
 * are passed through verbatim — only the documented ones below are interpreted.
 */
export interface OpenAIProviderMetadata {
	/** the responses-api item id (e.g. `rs_...`, `msg_...`, `fc_...`) for round-trip. */
	id?: string;
	/** opaque encrypted reasoning blob — required to resume reasoning models without server state. */
	encryptedContent?: string;
	/** reasoning-summary array as the responses api returned it. */
	summary?: unknown[];
	/** reasoning-content array as the responses api returned it. */
	content?: unknown[];
	/** item status from the responses api. */
	status?: 'in_progress' | 'completed' | 'incomplete';
}

/**
 * per-model overrides. add an entry here when a model has options the base shape doesn't cover. unlisted
 * models fall back to {@link OpenAIBaseProviderOptions}.
 */
interface OpenAIChatModelProviderOptionsByName {
	'gpt-5': OpenAIReasoningProviderOptions;
	'gpt-5-mini': OpenAIReasoningProviderOptions;
	'gpt-5-nano': OpenAIReasoningProviderOptions;
	o1: OpenAIReasoningProviderOptions;
	'o1-mini': OpenAIReasoningProviderOptions;
	o3: OpenAIReasoningProviderOptions;
	'o3-mini': OpenAIReasoningProviderOptions;
}

type ResolveProviderOptions<TModel extends string> = TModel extends keyof OpenAIChatModelProviderOptionsByName
	? OpenAIChatModelProviderOptionsByName[TModel]
	: OpenAIBaseProviderOptions;

// every internally-known field as optional, so we can read all of them off
// `providerOptions` regardless of which concrete shape callers passed.
type AnyOpenAIProviderOptions = OpenAIBaseProviderOptions & Partial<OpenAIReasoningProviderOptions>;

// #endregion

export interface OpenAIConfig {
	apiKey?: string;
	baseUrl?: string;
	headers?: Record<string, string>;
	fetch?: typeof fetch;
}

export type OpenAIAdapter<TModel extends OpenAIChatModel = OpenAIChatModel> = ChatAdapter<
	TModel,
	ResolveProviderOptions<TModel>
>;

/**
 * create an OpenAI chat adapter for a given model. provider options (e.g. `reasoningEffort`) narrow based on
 * the model literal.
 */
export const openai = <const TModel extends OpenAIChatModel>(
	model: TModel,
	config: OpenAIConfig = {},
): OpenAIAdapter<TModel> => {
	const apiKey = config.apiKey;
	const baseUrl = config.baseUrl ?? 'https://api.openai.com/v1';
	const fetcher = config.fetch ?? fetch;

	return createChatAdapter<OpenAIAdapter<TModel>>({
		kind: 'chat',
		provider: 'openai',
		model,
		chatStream(opts) {
			return streamOpenAI({ ...opts, model, apiKey, baseUrl, headers: config.headers, fetcher });
		},
		structuredOutput(opts) {
			return openaiStructuredOutput({
				...opts,
				model,
				apiKey,
				baseUrl,
				headers: config.headers,
				fetcher,
			});
		},
	});
};

// #region streaming impl

interface StreamOpenAIArgs extends ChatStreamOptions<AnyOpenAIProviderOptions> {
	model: string;
	apiKey: string | undefined;
	baseUrl: string;
	headers: Record<string, string> | undefined;
	fetcher: typeof fetch;
}

interface StreamToolCall {
	callId: string;
	name: string;
	args: string;
	ended: boolean;
}

interface StreamReasoning {
	itemId: string;
	streamId: string;
	open: boolean;
}

interface StreamText {
	itemId: string;
	streamId: string;
	open: boolean;
}

async function* streamOpenAI(args: StreamOpenAIArgs): AsyncGenerator<AdapterChunk, void, void> {
	const result = await postSse({
		url: `${args.baseUrl}/responses`,
		headers: {
			'content-type': 'application/json',
			...(args.apiKey ? { authorization: `Bearer ${args.apiKey}` } : {}),
			...args.headers,
		},
		body: JSON.stringify(buildRequest(args)),
		signal: args.signal,
		fetch: args.fetcher,
		errorPrefix: 'openai',
	});
	if (!result.ok) {
		yield result.error;
		return;
	}

	const toolsByItemId = new Map<string, StreamToolCall>();
	const reasoningByItemId = new Map<string, StreamReasoning>();
	const textByItemId = new Map<string, StreamText>();
	let finish: { reason: FinishReason; usage?: Usage } | undefined;

	const finalizeToolCall = async function* (
		entry: StreamToolCall,
		finalArgs: string,
		providerMetadata?: ProviderMetadata,
	): AsyncGenerator<AdapterChunk, void, void> {
		if (entry.ended) {
			return;
		}
		yield* emitMissingToolArgs(entry, finalArgs);
		entry.ended = true;
		yield { type: 'tool-call-end', id: entry.callId, name: entry.name, providerMetadata };
	};

	for await (const event of parseJsonSseStream<ResponseStreamEvent>(result.body, '[DONE]')) {
		switch (event.type) {
			case 'response.output_text.delta': {
				const text = textByItemId.get(event.item_id);
				if (text && event.delta) {
					yield { type: 'text-delta', id: text.streamId, delta: event.delta };
				}
				break;
			}
			case 'response.reasoning_text.delta':
			case 'response.reasoning_summary_text.delta': {
				const reasoning = reasoningByItemId.get(event.item_id);
				if (reasoning && event.delta) {
					yield { type: 'reasoning-delta', id: reasoning.streamId, delta: event.delta };
				}
				break;
			}
			case 'response.output_item.added': {
				const item = event.item;
				switch (item.type) {
					case 'function_call': {
						const itemId = item.id ?? item.call_id;
						const entry: StreamToolCall = {
							callId: item.call_id,
							name: item.name,
							args: '',
							ended: false,
						};
						toolsByItemId.set(itemId, entry);
						yield { type: 'tool-call-start', id: entry.callId, name: entry.name };
						break;
					}
					case 'reasoning': {
						const streamId = `r-${item.id}`;
						reasoningByItemId.set(item.id, { itemId: item.id, streamId, open: true });
						yield { type: 'reasoning-start', id: streamId };
						break;
					}
					case 'message': {
						const streamId = `t-${item.id}`;
						textByItemId.set(item.id, { itemId: item.id, streamId, open: true });
						yield { type: 'text-start', id: streamId };
						break;
					}
					default: {
						// unknown item types (server-tool calls, image generation, MCP, etc.)
						// fail fast rather than silently dropping — silent drops corrupt
						// replay and break prompt-cache prefixes.
						yield {
							type: 'error',
							error: new Error(`openai: unsupported output item type: ${item.type}`),
						};
						return;
					}
				}
				break;
			}
			case 'response.function_call_arguments.delta': {
				const entry = toolsByItemId.get(event.item_id);
				if (!entry || !event.delta) {
					break;
				}
				entry.args += event.delta;
				yield { type: 'tool-call-delta', id: entry.callId, name: entry.name, argsDelta: event.delta };
				break;
			}
			case 'response.function_call_arguments.done': {
				const entry = toolsByItemId.get(event.item_id);
				if (!entry) {
					break;
				}
				yield* finalizeToolCall(entry, event.arguments);
				finish = { reason: 'tool-calls', usage: finish?.usage };
				break;
			}
			case 'response.output_item.done': {
				const item = event.item;
				switch (item.type) {
					case 'function_call': {
						const itemId = item.id ?? item.call_id;
						const entry = toolsByItemId.get(itemId);
						if (entry) {
							const meta = openaiItemMetadata({ id: item.id, status: item.status });
							yield* finalizeToolCall(entry, item.arguments, meta);
							finish = { reason: 'tool-calls', usage: finish?.usage };
						}
						break;
					}
					case 'reasoning': {
						const reasoning = reasoningByItemId.get(item.id);
						if (reasoning?.open) {
							const meta = openaiItemMetadata({
								id: item.id,
								encryptedContent: item.encrypted_content ?? undefined,
								summary: item.summary,
								content: item.content,
								status: item.status,
							});
							reasoning.open = false;
							yield { type: 'reasoning-end', id: reasoning.streamId, providerMetadata: meta };
						}
						break;
					}
					case 'message': {
						const text = textByItemId.get(item.id);
						if (text?.open) {
							const meta = openaiItemMetadata({
								id: item.id,
								status: item.status,
								content: item.content,
							});
							text.open = false;
							yield { type: 'text-end', id: text.streamId, providerMetadata: meta };
						}
						break;
					}
				}
				break;
			}
			case 'response.completed': {
				finish = {
					reason: finish?.reason ?? mapResponseFinishReason(event.response),
					usage: toUsage(event.response.usage),
				};
				break;
			}
			case 'response.incomplete': {
				finish = {
					reason: mapResponseFinishReason(event.response),
					usage: toUsage(event.response.usage),
				};
				break;
			}
			case 'response.failed': {
				const message = event.response.error?.message ?? 'response failed';
				yield { type: 'error', error: new Error(`openai: ${message}`) };
				return;
			}
			case 'error': {
				yield { type: 'error', error: new Error(`openai: ${event.message}`) };
				return;
			}
		}
	}

	yield { type: 'finish', reason: finish?.reason ?? 'stop', usage: finish?.usage };
}

const openaiItemMetadata = (meta: OpenAIProviderMetadata): ProviderMetadata => ({ openai: meta });

const buildRequest = (args: StreamOpenAIArgs): ResponseCreateParamsStreaming => {
	const opts = args.providerOptions ?? {};
	const includeEncrypted = opts.includeEncryptedReasoning ?? opts.reasoningEffort !== undefined;

	const body: ResponseCreateParamsStreaming = {
		model: args.model,
		input: toOpenAIInput(args.messages),
		stream: true,
		temperature: args.temperature,
		top_p: args.topP,
		max_output_tokens: args.maxTokens,
		tools: args.tools && args.tools.length > 0 ? args.tools.map(toOpenAIWireTool) : undefined,
		user: opts.user,
		parallel_tool_calls: opts.parallelToolCalls,
		reasoning: opts.reasoningEffort ? { effort: opts.reasoningEffort } : undefined,
		text: opts.responseFormat ? { format: opts.responseFormat } : undefined,
		include: includeEncrypted ? ['reasoning.encrypted_content'] : undefined,
	};

	return body;
};

const toOpenAIInput = (messages: ModelMessage[]): ResponseInput => messages.flatMap(toOpenAIInputItems);

const readOpenAIMeta = (meta: ProviderMetadata | undefined): OpenAIProviderMetadata | undefined =>
	readProviderMeta<OpenAIProviderMetadata>(meta, 'openai');

const toOpenAIInputItems = (m: ModelMessage): ResponseInputItem[] => {
	switch (m.role) {
		case 'system': {
			const text = m.content.map((p) => p.text).join('');
			return [{ role: 'system', content: text, type: 'message' }];
		}
		case 'user': {
			const parts = m.content.map((p) => ({
				type: 'input_text' as const,
				text: p.text,
			}));
			return [{ role: 'user', content: parts, type: 'message' }];
		}
		case 'assistant': {
			return assistantToOpenAIItems(m.content);
		}
		case 'tool': {
			return m.content.map((part) => ({
				type: 'function_call_output' as const,
				call_id: part.toolCallId,
				output: part.output,
			}));
		}
		default: {
			const _exhaustive: never = m;
			void _exhaustive;
			// oxlint-disable-next-line typescript/no-unsafe-type-assertion
			throw new Error(`openai: unhandled message role: ${(m as ModelMessage).role}`);
		}
	}
};

const assistantToOpenAIItems = (content: AssistantContent): ResponseInputItem[] => {
	const out: ResponseInputItem[] = [];
	for (const part of content) {
		switch (part.type) {
			case 'reasoning': {
				const meta = readOpenAIMeta(part.providerMetadata);
				if (!meta?.id) {
					// without an id, the responses api can't accept the reasoning
					// item back. drop reasoning that wasn't produced by openai.
					break;
				}
				// oxlint-disable-next-line typescript/no-unsafe-type-assertion
				out.push({
					type: 'reasoning',
					id: meta.id,
					summary: meta.summary,
					content: meta.content,
					encrypted_content: meta.encryptedContent,
					status: meta.status,
				} as unknown as ResponseInputItem);
				break;
			}
			case 'redacted-reasoning': {
				// openai has no analogue; ignore. (this part exists for anthropic.)
				break;
			}
			case 'text': {
				const meta = readOpenAIMeta(part.providerMetadata);
				// oxlint-disable-next-line typescript/no-unsafe-type-assertion
				out.push({
					type: 'message',
					role: 'assistant',
					id: meta?.id,
					status: meta?.status,
					content: meta?.content ?? [{ type: 'output_text', text: part.text, annotations: [] }],
				} as unknown as ResponseInputItem);
				break;
			}
			case 'tool-call': {
				const meta = readOpenAIMeta(part.providerMetadata);
				// oxlint-disable-next-line typescript/no-unsafe-type-assertion
				out.push({
					type: 'function_call',
					id: meta?.id,
					call_id: part.id,
					name: part.name,
					arguments: part.arguments,
				} as unknown as ResponseInputItem);
				break;
			}
		}
	}
	return out;
};

const toOpenAIWireTool = (t: WireTool): FunctionTool => {
	return {
		type: 'function',
		name: t.name,
		description: t.description,
		parameters: t.jsonSchema,
		strict: null,
	};
};

async function* emitMissingToolArgs(
	entry: StreamToolCall,
	argumentsText: string,
): AsyncGenerator<AdapterChunk, void, void> {
	if (argumentsText === entry.args) {
		return;
	}
	if (argumentsText.startsWith(entry.args)) {
		const delta = argumentsText.slice(entry.args.length);
		entry.args = argumentsText;
		if (delta.length > 0) {
			yield { type: 'tool-call-delta', id: entry.callId, name: entry.name, argsDelta: delta };
		}
		return;
	}
	// canonical args diverged from concatenated deltas (rare). resync silently;
	// emitting the full text as another delta would double-feed downstream.
	entry.args = argumentsText;
}

const mapResponseFinishReason = (response: Response): FinishReason => {
	if (response.status === 'incomplete') {
		if (response.incomplete_details?.reason === 'max_output_tokens') {
			return 'length';
		}
		if (response.incomplete_details?.reason === 'content_filter') {
			return 'content-filter';
		}
	}
	if (response.status === 'failed') {
		return 'error';
	}
	return 'stop';
};

const toUsage = (usage: Response['usage']): Usage | undefined => {
	if (!usage) {
		return undefined;
	}
	return {
		inputTokens: usage.input_tokens,
		outputTokens: usage.output_tokens,
		cacheReadInputTokens: usage.input_tokens_details?.cached_tokens,
		reasoningTokens: usage.output_tokens_details?.reasoning_tokens,
	};
};

// #endregion

// #region structured output

interface StructuredOpenAIArgs extends StructuredOutputOptions<AnyOpenAIProviderOptions> {
	model: string;
	apiKey: string | undefined;
	baseUrl: string;
	headers: Record<string, string> | undefined;
	fetcher: typeof fetch;
}

const openaiStructuredOutput = async (args: StructuredOpenAIArgs): Promise<StructuredOutputResult> => {
	const strictSchema = makeOpenAIStrictCompatible(args.jsonSchema);

	const body: ResponseCreateParamsNonStreaming = {
		model: args.model,
		input: toOpenAIInput(args.messages),
		stream: false,
		text: {
			format: {
				type: 'json_schema',
				name: args.name ?? 'output',
				schema: strictSchema,
				strict: true,
			},
		},
		temperature: args.temperature,
		top_p: args.topP,
		max_output_tokens: args.maxTokens,
	};

	const json = await postJson<Response>({
		url: `${args.baseUrl}/responses`,
		headers: {
			'content-type': 'application/json',
			...(args.apiKey ? { authorization: `Bearer ${args.apiKey}` } : {}),
			...args.headers,
		},
		body: JSON.stringify(body),
		signal: args.signal,
		fetch: args.fetcher,
		errorPrefix: 'openai',
	});
	const rawText = json.output_text ?? '';

	let parsed: unknown;
	try {
		parsed = JSON.parse(rawText);
	} catch (e) {
		throw new Error('openai: structured output response was not valid json', { cause: e });
	}

	return { data: stripNulls(parsed), rawText };
};

// #endregion
