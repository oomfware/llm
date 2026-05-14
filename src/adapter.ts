import type { AdapterChunk, ModelMessage } from './types.ts';

/**
 * a tool as the adapter sees it on the wire. JSON Schema has already been extracted from the tool's Standard
 * JSON Schema by `chat()`, so adapters never deal with schema libs.
 */
export interface WireTool {
	name: string;
	description?: string;
	jsonSchema: Record<string, unknown>;
}

/** what `chat()` hands to an adapter on each turn. */
export interface ChatStreamOptions<TProviderOptions> {
	messages: ModelMessage[];
	tools?: WireTool[];
	temperature?: number;
	topP?: number;
	maxTokens?: number;
	signal?: AbortSignal;
	providerOptions?: TProviderOptions;
}

/** options passed to {@link ChatAdapter.structuredOutput}. */
export interface StructuredOutputOptions<TProviderOptions> {
	messages: ModelMessage[];
	/** JSON Schema for the expected output, already extracted from a Standard JSON Schema. */
	jsonSchema: Record<string, unknown>;
	/** optional schema name (used by some providers as a label, e.g. OpenAI's `json_schema.name`). */
	name?: string;
	temperature?: number;
	topP?: number;
	maxTokens?: number;
	signal?: AbortSignal;
	providerOptions?: TProviderOptions;
}

export interface StructuredOutputResult {
	/** parsed JSON from the model — not yet schema-validated. */
	data: unknown;
	/** raw response text the JSON was extracted from. */
	rawText: string;
}

/**
 * a chat adapter. created by a provider factory like `openai('gpt-4o')`, which pre-resolves `TModel` and
 * `TProviderOptions` so callers never write generics by hand.
 *
 * the `~types` property is a phantom: it carries inference info to `chat()` via
 * `TAdapter['~types']['providerOptions']` but is never assigned at runtime.
 */
export interface ChatAdapter<TModel extends string = string, TProviderOptions = unknown> {
	readonly kind: 'chat';
	readonly provider: string;
	readonly model: TModel;
	readonly '~types': {
		providerOptions: TProviderOptions;
	};
	chatStream(options: ChatStreamOptions<TProviderOptions>): AsyncIterable<AdapterChunk>;
	/**
	 * make a single non-streaming model call constrained to a JSON Schema. called by `generateObject()` after
	 * the agent loop converges. providers that don't natively support structured output may simulate it (e.g.
	 * via tool prefill on Anthropic).
	 */
	structuredOutput(options: StructuredOutputOptions<TProviderOptions>): Promise<StructuredOutputResult>;
}

export type AnyChatAdapter = ChatAdapter<any, any>;

/**
 * runtime constructor for adapter values. lets provider factories return a plain object without having to
 * fake the phantom `~types` field at runtime.
 *
 * the cast is the single isolated point where the phantom field is fabricated.
 */
export const createChatAdapter = <T extends AnyChatAdapter>(adapter: Omit<T, '~types'>): T =>
	// oxlint-disable-next-line typescript/no-unsafe-type-assertion
	adapter as T;
