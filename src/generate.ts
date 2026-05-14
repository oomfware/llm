import type { AnyChatAdapter } from './adapter.ts';
import { chat, type ChatOptions } from './chat.ts';
import type { AnyTool } from './tool.ts';
import type { FinishReason, ModelMessage, ToolCallPart, Usage } from './types.ts';

export interface ToolResultRecord {
	id: string;
	name: string;
	result: unknown;
}

export interface GenerateResult {
	/** all assistant text concatenated across the agent loop. */
	text: string;
	/** reasoning/thinking content concatenated across the loop, if the model produced any. */
	reasoning: string;
	/** every tool call the model made across all iterations. */
	toolCalls: ToolCallPart[];
	/** every tool result executed during the loop, in submission order. */
	toolResults: ToolResultRecord[];
	/** the full conversation: input messages + every assistant turn + every tool message. */
	messages: ModelMessage[];
	finishReason: FinishReason;
	usage: Usage | undefined;
}

/**
 * run an agentic chat turn and return everything when it finishes.
 *
 * for streaming UIs use {@link chat} instead — this drains the stream into a single result object.
 *
 * @throws whatever the adapter or any tool throws (re-thrown from `error` chunks).
 */
export const generate = async <
	TAdapter extends AnyChatAdapter,
	const TTools extends Record<string, AnyTool> = {},
>(
	options: ChatOptions<TAdapter, TTools>,
): Promise<GenerateResult> => {
	const messages: ModelMessage[] = [...options.messages];
	let text = '';
	let reasoning = '';
	const toolCalls: ToolCallPart[] = [];
	const toolResults: ToolResultRecord[] = [];
	let finishReason: FinishReason = 'stop';
	let usage: Usage | undefined;

	for await (const chunk of chat(options)) {
		switch (chunk.type) {
			case 'text-delta': {
				text += chunk.delta;
				break;
			}
			case 'reasoning-delta': {
				reasoning += chunk.delta;
				break;
			}
			case 'message': {
				messages.push(chunk.message);
				if (chunk.message.role === 'assistant') {
					for (const part of chunk.message.content) {
						if (part.type === 'tool-call') {
							toolCalls.push(part);
						}
					}
				}
				break;
			}
			case 'tool-result': {
				toolResults.push({ id: chunk.id, name: chunk.name, result: chunk.result });
				break;
			}
			case 'finish': {
				finishReason = chunk.reason;
				usage = chunk.usage;
				break;
			}
			case 'error': {
				throw chunk.error;
			}
			default: {
				break;
			}
		}
	}

	return { text, reasoning, toolCalls, toolResults, messages, finishReason, usage };
};
