import type { InferToolInput, InferToolOutput } from './tool.ts';

export type Role = 'system' | 'user' | 'assistant' | 'tool';

/**
 * a free-form bag of provider-specific data that rides along on messages and parts. keyed by provider name
 * (e.g. `anthropic`, `openai`); the values are defined by each adapter and are read+written without core type
 * checks.
 *
 * the bag exists so byte-faithful round-trips work without polluting the core types with provider-specific
 * fields. for example, `providerMetadata.anthropic.signature` carries an extended-thinking signature, and
 * `providerMetadata.anthropic.cacheControl` marks a prompt-caching breakpoint. see each adapter for its own
 * conventions.
 */
export type ProviderMetadata = Record<string, unknown>;

// #region content parts

export interface TextPart {
	type: 'text';
	text: string;
	providerMetadata?: ProviderMetadata;
}

/**
 * a chunk of model reasoning ("thinking") content. opaque verification data that providers require on resume
 * (e.g. anthropic's signature) lives in `providerMetadata` and is round-tripped verbatim by the relevant
 * adapter.
 */
export interface ReasoningPart {
	type: 'reasoning';
	text: string;
	providerMetadata?: ProviderMetadata;
}

/**
 * an opaque reasoning block — anthropic returns these when its safety classifier replaces a thinking block.
 * the `data` payload lives in `providerMetadata.anthropic.data` and must be returned unchanged on resume;
 * there is no human-readable text.
 */
export interface RedactedReasoningPart {
	type: 'redacted-reasoning';
	providerMetadata?: ProviderMetadata;
}

/**
 * a tool call as produced by the model. arguments are kept as a json string because that's what every
 * provider streams; parsing is deferred to the agent loop right before invocation.
 *
 * `approval` carries the user's decision for tools marked `needsApproval: true`. it lives on the persisted
 * assistant message so that history naturally carries the decision: present + `approved: true` runs the call
 * on resume, present + `approved: false` injects a rejection tool message, absent suspends the loop with
 * reason `awaiting-approval`. adapters do not serialize this field — it's read only by the agent loop.
 */
export interface ToolCallPart {
	type: 'tool-call';
	id: string;
	name: string;
	arguments: string;
	approval?: { approved: boolean };
	providerMetadata?: ProviderMetadata;
}

export interface ToolResultPart {
	type: 'tool-result';
	toolCallId: string;
	output: string;
	/** when true, signals to the model that the tool execution failed (e.g. anthropic's `is_error: true`). */
	isError?: boolean;
	providerMetadata?: ProviderMetadata;
}

// #endregion

// #region per-role content types

/** content allowed on a system message — text only. */
export type SystemContent = TextPart[];

/** content allowed on a user message — text only (multimodal parts may be added later). */
export type UserContent = TextPart[];

/**
 * content allowed on an assistant message — text, reasoning, redacted reasoning, and tool calls, in the exact
 * order the model produced them.
 *
 * preserving this order matters for prompt caching: anthropic and openai both compute cache prefixes against
 * the wire bytes, so reordering parts between turns invalidates the cache.
 */
export type AssistantContent = (TextPart | ReasoningPart | RedactedReasoningPart | ToolCallPart)[];

/** content of a tool turn — one or more tool results, possibly batched. */
export type ToolContent = ToolResultPart[];

// #endregion

// #region messages

export interface SystemMessage {
	role: 'system';
	content: SystemContent;
	providerMetadata?: ProviderMetadata;
}

export interface UserMessage {
	role: 'user';
	content: UserContent;
	providerMetadata?: ProviderMetadata;
}

export interface AssistantMessage {
	role: 'assistant';
	content: AssistantContent;
	providerMetadata?: ProviderMetadata;
}

export interface ToolMessage {
	role: 'tool';
	content: ToolContent;
	providerMetadata?: ProviderMetadata;
}

export type ModelMessage = SystemMessage | UserMessage | AssistantMessage | ToolMessage;

// #endregion

export type FinishReason =
	| 'stop'
	| 'length'
	| 'tool-calls'
	| 'content-filter'
	| 'error'
	/**
	 * the loop suspended because at least one tool call in the most recent assistant turn is gated by
	 * `needsApproval` and no decision was supplied. resume by setting `approval` on each pending tool-call part
	 * in the persisted assistant message and re-invoking `chat()` with the same `messages` array.
	 */
	| 'awaiting-approval';

export interface Usage {
	inputTokens: number;
	outputTokens: number;
	/** prompt-cache writes (anthropic `cache_creation_input_tokens`). */
	cacheCreationInputTokens?: number;
	/** prompt-cache hits (anthropic `cache_read_input_tokens` / openai `cached_tokens`). */
	cacheReadInputTokens?: number;
	/** reasoning tokens included in `outputTokens` (openai `output_tokens_details.reasoning_tokens`). */
	reasoningTokens?: number;
}

// #region adapter-emitted chunks

/**
 * what an adapter yields on the wire. the agent loop in `chat()` consumes these, accumulates parts, runs
 * tools, and re-emits a richer {@link StreamChunk} for the public api.
 *
 * parts are opened (`*-start`), streamed in deltas, then closed (`*-end`). the chat loop builds the assistant
 * message's `content[]` by appending each closed part in the order they end, which is the order the model
 * emitted them on the wire.
 */
export type AdapterChunk =
	| { type: 'text-start'; id: string; providerMetadata?: ProviderMetadata }
	| { type: 'text-delta'; id: string; delta: string }
	| { type: 'text-end'; id: string; providerMetadata?: ProviderMetadata }
	| { type: 'reasoning-start'; id: string; providerMetadata?: ProviderMetadata }
	| { type: 'reasoning-delta'; id: string; delta: string }
	| { type: 'reasoning-end'; id: string; providerMetadata?: ProviderMetadata }
	| { type: 'redacted-reasoning'; providerMetadata: ProviderMetadata }
	| { type: 'tool-call-start'; name: string; id: string; providerMetadata?: ProviderMetadata }
	| { type: 'tool-call-delta'; name: string; id: string; argsDelta: string }
	| { type: 'tool-call-end'; name: string; id: string; providerMetadata?: ProviderMetadata }
	| { type: 'finish'; reason: FinishReason; usage?: Usage; providerMetadata?: ProviderMetadata }
	| { type: 'error'; error: unknown };

// #endregion

// #region public stream chunks (generic over registered tools)

type ToolName<TTools> = keyof TTools & string;

/**
 * the tool-related variants of {@link StreamChunk}, distributed over each registered tool so `name` is a
 * literal and `input`/`result` narrow.
 *
 * collapses to `never` when `TTools` is empty.
 */
export type ToolChunks<TTools> = {
	[K in ToolName<TTools>]:
		| { type: 'tool-call-start'; name: K; id: string }
		| { type: 'tool-call-delta'; name: K; id: string; argsDelta: string }
		| { type: 'tool-call-end'; name: K; id: string; input: InferToolInput<TTools[K]> }
		| { type: 'tool-approval-requested'; name: K; id: string; input: InferToolInput<TTools[K]> }
		| { type: 'tool-rejected'; name: K; id: string; reason: string }
		| { type: 'tool-result'; name: K; id: string; result: InferToolOutput<TTools[K]> };
}[ToolName<TTools>];

/**
 * the public chunk union yielded by `chat()`. parameterised by the tools record so that:
 *
 * - `chunk.name` on tool variants is the literal key (e.g. `'get_weather'`)
 * - discriminating on `chunk.name` narrows `chunk.input` / `chunk.result` to the matching tool's input/output
 *   types
 *
 * with no tools registered (`TTools = {}`), tool variants drop out entirely.
 *
 * a `message` chunk is yielded each time the loop commits an assistant or tool message to the conversation.
 * callers persisting history can append `chunk.message` directly; on `awaiting-approval`, the most recent
 * assistant `message` chunk is the resume handle whose tool-call parts' `approval` field the caller mutates
 * before re-invoking `chat()`.
 */
export type StreamChunk<TTools = {}> =
	| { type: 'text-delta'; delta: string }
	| { type: 'reasoning-delta'; delta: string }
	| ToolChunks<TTools>
	| { type: 'message'; message: AssistantMessage | ToolMessage }
	| { type: 'finish'; reason: FinishReason; usage?: Usage }
	| { type: 'error'; error: unknown };

// #endregion
