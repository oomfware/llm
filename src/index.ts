export { chat, type ChatOptions } from './chat.ts';
export { generate, type GenerateResult, type ToolResultRecord } from './generate.ts';
export {
	generateObject,
	type GenerateObjectOptions,
	type GenerateObjectResult,
	type ObjectSchema,
} from './generate-object.ts';
export {
	combineStrategies,
	maxIterations,
	untilFinishReason,
	type AgentLoopState,
	type AgentLoopStrategy,
} from './agent-loop-strategies.ts';
export {
	tool,
	type AnyTool,
	type InferToolInput,
	type InferToolOutput,
	type Tool,
	type ToolContext,
	type ToolSchema,
} from './tool.ts';
export { jsonSchema } from './json-schema.ts';
export { assistant, system, text, toolResults, user } from './messages.ts';
export {
	AIError,
	StructuredOutputValidationError,
	ToolArgumentsParseError,
	ToolInputValidationError,
	ToolOutputValidationError,
	UnknownToolError,
} from './errors.ts';
export { smoothStream, type SmoothChunking, type SmoothStreamOptions } from './smooth-stream.ts';
export {
	createChatAdapter,
	type AnyChatAdapter,
	type ChatAdapter,
	type ChatStreamOptions,
	type StructuredOutputOptions,
	type StructuredOutputResult,
	type WireTool,
} from './adapter.ts';
export type {
	AdapterChunk,
	AssistantContent,
	AssistantMessage,
	FinishReason,
	ModelMessage,
	ProviderMetadata,
	ReasoningPart,
	RedactedReasoningPart,
	Role,
	StreamChunk,
	SystemContent,
	SystemMessage,
	TextPart,
	ToolCallPart,
	ToolChunks,
	ToolContent,
	ToolMessage,
	ToolResultPart,
	UserContent,
	Usage,
	UserMessage,
} from './types.ts';

export {
	dummy,
	type DummyAdapter,
	type DummyConfig,
	type DummyPart,
	type DummyResponse,
} from './providers/dummy.ts';
export {
	openai,
	type OpenAIAdapter,
	type OpenAIChatModel,
	type OpenAIConfig,
	type OpenAIProviderMetadata,
} from './providers/openai.ts';
export {
	openaiCompatible,
	type OpenAICompatibleConfig,
	type OpenAICompatibleProviderOptions,
} from './providers/openai-compatible.ts';
export {
	openrouter,
	type OpenRouterAdapter,
	type OpenRouterConfig,
	type OpenRouterModel,
	type OpenRouterProviderOptions,
	type OpenRouterProviderPreferences,
} from './providers/openrouter.ts';
export {
	anthropic,
	type AnthropicAdapter,
	type AnthropicCacheControl,
	type AnthropicConfig,
	type AnthropicModel,
	type AnthropicProviderMetadata,
} from './providers/anthropic.ts';
