import type {
	ChatCompletion,
	ChatCompletionAssistantMessageParam,
	ChatCompletionChunk,
	ChatCompletionContentPartText,
	ChatCompletionCreateParamsNonStreaming,
	ChatCompletionCreateParamsStreaming,
	ChatCompletionFunctionTool,
	ChatCompletionMessageParam,
} from 'openai/resources/chat/completions';

import {
	createChatAdapter,
	type ChatAdapter,
	type ChatStreamOptions,
	type StructuredOutputOptions,
	type StructuredOutputResult,
	type WireTool,
} from '../adapter.ts';
import { parseJsonSseStream, postSse } from '../internal/http.ts';
import { makeOpenAIStrictCompatible, stripNulls } from '../internal/openai-strict.ts';
import type { AdapterChunk, FinishReason, ModelMessage, Usage } from '../types.ts';

/**
 * provider options shared by all OpenAI Chat Completions-shaped backends. wrappers (e.g. `openrouter`) extend
 * this with their own fields.
 */
export interface OpenAICompatibleProviderOptions {
	user?: string;
	parallelToolCalls?: boolean;
	seed?: number;
	reasoningEffort?: 'low' | 'medium' | 'high';
	responseFormat?: { type: 'json_object' } | { type: 'json_schema'; jsonSchema: unknown };
}

export interface OpenAICompatibleConfig {
	/**
	 * provider label. surfaces on the returned adapter as `adapter.provider` and is used in error message
	 * prefixes.
	 */
	name: string;
	/** base url for the chat completions endpoint, without `/chat/completions`. */
	baseUrl: string;
	apiKey?: string;
	headers?: Record<string, string>;
	fetch?: typeof fetch;
	/**
	 * hook for wrappers to fold provider-specific provider-options into the wire body. receives the raw
	 * `providerOptions` value and returns extra fields to merge into the request.
	 */
	extendBody?: (providerOptions: Record<string, unknown>) => Record<string, unknown>;
}

/**
 * create a chat adapter against an OpenAI Chat Completions-compatible endpoint. use this directly for ad-hoc
 * compat providers, or wrap it (see `openrouter`) to expose typed model literals and provider-specific
 * options.
 */
export const openaiCompatible = <
	const TModel extends string,
	TOptions extends OpenAICompatibleProviderOptions = OpenAICompatibleProviderOptions,
>(
	model: TModel,
	config: OpenAICompatibleConfig,
): ChatAdapter<TModel, TOptions> => {
	const apiKey = config.apiKey;
	const fetcher = config.fetch ?? fetch;

	return createChatAdapter<ChatAdapter<TModel, TOptions>>({
		kind: 'chat',
		provider: config.name,
		model,
		chatStream(opts) {
			return streamChatCompletions({
				...opts,
				name: config.name,
				model,
				apiKey,
				baseUrl: config.baseUrl,
				headers: config.headers,
				fetcher,
				extendBody: config.extendBody,
			});
		},
		structuredOutput(opts) {
			return chatCompletionsStructuredOutput({
				...opts,
				name: config.name,
				model,
				apiKey,
				baseUrl: config.baseUrl,
				headers: config.headers,
				fetcher,
				extendBody: config.extendBody,
			});
		},
	});
};

// #region streaming impl

interface StreamArgs extends ChatStreamOptions<OpenAICompatibleProviderOptions> {
	name: string;
	model: string;
	apiKey: string | undefined;
	baseUrl: string;
	headers: Record<string, string> | undefined;
	fetcher: typeof fetch;
	extendBody: ((opts: Record<string, unknown>) => Record<string, unknown>) | undefined;
}

const TEXT_STREAM_ID = 't';

async function* streamChatCompletions(args: StreamArgs): AsyncGenerator<AdapterChunk, void, void> {
	const result = await postSse({
		url: `${args.baseUrl}/chat/completions`,
		headers: {
			'content-type': 'application/json',
			...(args.apiKey ? { authorization: `Bearer ${args.apiKey}` } : {}),
			...args.headers,
		},
		body: JSON.stringify(buildRequest(args)),
		signal: args.signal,
		fetch: args.fetcher,
		errorPrefix: args.name,
	});
	if (!result.ok) {
		yield result.error;
		return;
	}

	// chat completions doesn't natively bracket text the way responses does;
	// fabricate a single text-start/text-end pair around any text deltas.
	let textOpen = false;

	// index-keyed tool-call accumulator. OpenAI streams partial tool calls
	// keyed by an integer `index`. usually `id` and `function.name` arrive
	// on the first delta, but we don't rely on that — we buffer args until
	// both are known and only then emit `tool-call-start` + flush.
	const toolByIndex = new Map<
		number,
		{
			id: string | undefined;
			name: string | undefined;
			emittedStart: boolean;
			bufferedArgs: string;
		}
	>();
	let finish: { reason: FinishReason; usage?: Usage } | undefined;

	for await (const event of parseJsonSseStream<ChatCompletionChunk>(result.body, '[DONE]')) {
		const choice = event.choices?.[0];
		if (!choice) {
			if (event.usage) {
				finish = {
					reason: finish?.reason ?? 'stop',
					usage: { inputTokens: event.usage.prompt_tokens, outputTokens: event.usage.completion_tokens },
				};
			}
			continue;
		}

		const delta = choice.delta;

		if (delta?.content) {
			if (!textOpen) {
				yield { type: 'text-start', id: TEXT_STREAM_ID };
				textOpen = true;
			}
			yield { type: 'text-delta', id: TEXT_STREAM_ID, delta: delta.content };
		}

		if (delta?.tool_calls) {
			for (const tc of delta.tool_calls) {
				let entry = toolByIndex.get(tc.index);
				if (!entry) {
					entry = {
						id: tc.id,
						name: tc.function?.name,
						emittedStart: false,
						bufferedArgs: '',
					};
					toolByIndex.set(tc.index, entry);
				} else {
					if (!entry.id && tc.id) {
						entry.id = tc.id;
					}
					if (!entry.name && tc.function?.name) {
						entry.name = tc.function.name;
					}
				}

				if (!entry.emittedStart && entry.id && entry.name) {
					yield { type: 'tool-call-start', id: entry.id, name: entry.name };
					entry.emittedStart = true;
					if (entry.bufferedArgs.length > 0) {
						yield {
							type: 'tool-call-delta',
							id: entry.id,
							name: entry.name,
							argsDelta: entry.bufferedArgs,
						};
						entry.bufferedArgs = '';
					}
				}

				if (tc.function?.arguments) {
					if (entry.emittedStart && entry.id && entry.name) {
						yield {
							type: 'tool-call-delta',
							id: entry.id,
							name: entry.name,
							argsDelta: tc.function.arguments,
						};
					} else {
						entry.bufferedArgs += tc.function.arguments;
					}
				}
			}
		}

		if (choice.finish_reason) {
			finish = {
				reason: mapFinishReason(choice.finish_reason),
				usage: finish?.usage,
			};
		}
	}

	if (textOpen) {
		yield { type: 'text-end', id: TEXT_STREAM_ID };
	}

	for (const entry of toolByIndex.values()) {
		if (entry.emittedStart && entry.id && entry.name) {
			yield { type: 'tool-call-end', id: entry.id, name: entry.name };
		}
	}

	yield { type: 'finish', reason: finish?.reason ?? 'stop', usage: finish?.usage };
}

const buildRequest = (args: StreamArgs): ChatCompletionCreateParamsStreaming => {
	const opts = args.providerOptions ?? {};

	const body: ChatCompletionCreateParamsStreaming = {
		model: args.model,
		messages: args.messages.flatMap(toChatCompletionsMessages),
		stream: true,
		stream_options: { include_usage: true },
		temperature: args.temperature,
		top_p: args.topP,
		max_completion_tokens: args.maxTokens,
		tools: args.tools && args.tools.length > 0 ? args.tools.map(toChatCompletionsWireTool) : undefined,
		user: opts.user,
		parallel_tool_calls: opts.parallelToolCalls,
		seed: opts.seed,
		reasoning_effort: opts.reasoningEffort,
	};

	if (opts.responseFormat) {
		body.response_format =
			opts.responseFormat.type === 'json_schema'
				? // oxlint-disable-next-line typescript/no-unsafe-type-assertion
					{ type: 'json_schema', json_schema: opts.responseFormat.jsonSchema as { name: string } }
				: opts.responseFormat;
	}

	if (args.extendBody) {
		// oxlint-disable-next-line typescript/no-unsafe-type-assertion
		Object.assign(body, args.extendBody(opts as Record<string, unknown>));
	}

	return body;
};

const toChatCompletionsMessages = (m: ModelMessage): ChatCompletionMessageParam[] => {
	switch (m.role) {
		case 'system': {
			const text = m.content.map((p) => p.text).join('');
			return [{ role: 'system', content: text }];
		}
		case 'user': {
			const parts: ChatCompletionContentPartText[] = m.content.map((p) => ({
				type: 'text',
				text: p.text,
			}));
			return [{ role: 'user', content: parts }];
		}
		case 'assistant': {
			const out: ChatCompletionAssistantMessageParam = { role: 'assistant' };
			let text = '';
			const toolCalls: NonNullable<ChatCompletionAssistantMessageParam['tool_calls']> = [];
			for (const part of m.content) {
				switch (part.type) {
					case 'text': {
						text += part.text;
						break;
					}
					case 'tool-call': {
						toolCalls.push({
							id: part.id,
							type: 'function',
							function: { name: part.name, arguments: part.arguments },
						});
						break;
					}
					// reasoning and redacted-reasoning don't round-trip through chat
					// completions; the api doesn't accept thinking blocks back.
				}
			}
			if (text) {
				out.content = text;
			}
			if (toolCalls.length > 0) {
				out.tool_calls = toolCalls;
			}
			return [out];
		}
		case 'tool': {
			// chat completions sends one wire message per tool result.
			return m.content.map((part) => ({
				role: 'tool',
				tool_call_id: part.toolCallId,
				content: part.output,
			}));
		}
		default: {
			const _exhaustive: never = m;
			void _exhaustive;
			// oxlint-disable-next-line typescript/no-unsafe-type-assertion
			throw new Error(`openai-compatible: unhandled message role: ${(m as ModelMessage).role}`);
		}
	}
};

const toChatCompletionsWireTool = (t: WireTool): ChatCompletionFunctionTool => {
	return {
		type: 'function',
		function: {
			name: t.name,
			description: t.description,
			parameters: t.jsonSchema,
		},
	};
};

const mapFinishReason = (reason: string): FinishReason => {
	switch (reason) {
		case 'stop':
			return 'stop';
		case 'length':
			return 'length';
		case 'tool_calls':
		case 'function_call':
			return 'tool-calls';
		case 'content_filter':
			return 'content-filter';
		default:
			return 'stop';
	}
};

// #endregion

// #region structured output

interface StructuredArgs extends StructuredOutputOptions<OpenAICompatibleProviderOptions> {
	name: string;
	model: string;
	apiKey: string | undefined;
	baseUrl: string;
	headers: Record<string, string> | undefined;
	fetcher: typeof fetch;
	extendBody: ((opts: Record<string, unknown>) => Record<string, unknown>) | undefined;
}

const chatCompletionsStructuredOutput = async (args: StructuredArgs): Promise<StructuredOutputResult> => {
	const strictSchema = makeOpenAIStrictCompatible(args.jsonSchema);

	const body: ChatCompletionCreateParamsNonStreaming = {
		model: args.model,
		messages: args.messages.flatMap(toChatCompletionsMessages),
		stream: false,
		response_format: {
			type: 'json_schema',
			json_schema: {
				name: args.name ?? 'output',
				schema: strictSchema,
				strict: true,
			},
		},
		temperature: args.temperature,
		top_p: args.topP,
		max_completion_tokens: args.maxTokens,
	};

	if (args.extendBody) {
		// oxlint-disable-next-line typescript/no-unsafe-type-assertion
		Object.assign(body, args.extendBody((args.providerOptions ?? {}) as Record<string, unknown>));
	}

	const response = await args.fetcher(`${args.baseUrl}/chat/completions`, {
		method: 'POST',
		headers: {
			'content-type': 'application/json',
			...(args.apiKey ? { authorization: `Bearer ${args.apiKey}` } : {}),
			...args.headers,
		},
		body: JSON.stringify(body),
		signal: args.signal,
	});

	if (!response.ok) {
		const text = await response.text().catch(() => '');
		throw new Error(`${args.name}: ${response.status} ${response.statusText} ${text}`);
	}

	// oxlint-disable-next-line typescript/no-unsafe-type-assertion
	const json = (await response.json()) as ChatCompletion;
	const rawText = json.choices?.[0]?.message?.content ?? '';

	let parsed: unknown;
	try {
		parsed = JSON.parse(rawText);
	} catch (e) {
		throw new Error(`${args.name}: structured output response was not valid json`, { cause: e });
	}

	return { data: stripNulls(parsed), rawText };
};

// #endregion
