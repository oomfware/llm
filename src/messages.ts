import type {
	AssistantContent,
	AssistantMessage,
	ProviderMetadata,
	SystemMessage,
	TextPart,
	ToolContent,
	ToolMessage,
	ToolResultPart,
	UserMessage,
} from './types.ts';

interface MetaOptions {
	providerMetadata?: ProviderMetadata;
}

/**
 * build a {@link TextPart}.
 *
 * @example
 * 	text('hello');
 * 	text('cached', { providerMetadata: { anthropic: { cacheControl: { type: 'ephemeral' } } } });
 */
export const text = (content: string, options: MetaOptions = {}): TextPart => ({
	type: 'text',
	text: content,
	providerMetadata: options.providerMetadata,
});

/**
 * build a {@link SystemMessage} from a string.
 *
 * @example
 * 	system('answer in one sentence.');
 */
export const system = (content: string, options: MetaOptions = {}): SystemMessage => ({
	role: 'system',
	content: [{ type: 'text', text: content }],
	providerMetadata: options.providerMetadata,
});

/**
 * build a {@link UserMessage}. accepts either a plain string or a content array for advanced cases (e.g.
 * attaching a cache-control marker to a specific text part).
 *
 * @example
 * 	user('what is sqlite?');
 * 	user([
 * 		text('part one'),
 * 		text('part two', {
 * 			providerMetadata: { anthropic: { cacheControl: { type: 'ephemeral' } } },
 * 		}),
 * 	]);
 */
export const user = (content: string | TextPart[], options: MetaOptions = {}): UserMessage => ({
	role: 'user',
	content: typeof content === 'string' ? [{ type: 'text', text: content }] : content,
	providerMetadata: options.providerMetadata,
});

/**
 * build an {@link AssistantMessage}. mostly useful in tests and when pre-seeding the conversation with a
 * prefilled assistant turn.
 */
export const assistant = (
	content: string | AssistantContent,
	options: MetaOptions = {},
): AssistantMessage => ({
	role: 'assistant',
	content: typeof content === 'string' ? [{ type: 'text', text: content }] : content,
	providerMetadata: options.providerMetadata,
});

/**
 * build a {@link ToolMessage} carrying one or more tool results.
 *
 * @example
 * 	toolResults([
 * 		{ toolCallId: 'call_1', output: '{"temperatureC":23}' },
 * 		{ toolCallId: 'call_2', output: 'error: not found', isError: true },
 * 	]);
 */
export const toolResults = (
	results: ToolContent | ToolResultPart[],
	options: MetaOptions = {},
): ToolMessage => ({
	role: 'tool',
	content: results,
	providerMetadata: options.providerMetadata,
});
