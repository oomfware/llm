import type { AnyChatAdapter, WireTool } from './adapter.ts';
import { maxIterations as maxIterationsStrategy } from './agent-loop-strategies.ts';
import type { AgentLoopState, AgentLoopStrategy } from './agent-loop-strategies.ts';
import {
	ToolArgumentsParseError,
	ToolInputValidationError,
	ToolOutputValidationError,
	UnknownToolError,
} from './errors.ts';
import type { AnyTool, ToolSchema } from './tool.ts';
import type {
	AssistantContent,
	AssistantMessage,
	FinishReason,
	ModelMessage,
	ProviderMetadata,
	StreamChunk,
	ToolCallPart,
	ToolMessage,
	ToolResultPart,
	Usage,
} from './types.ts';

/**
 * options for {@link chat}. provider-specific options are typed against the
 * adapter's pre-resolved `~types.providerOptions`, so passing
 * `openai('gpt-5')` narrows `providerOptions` to gpt-5's shape.
 */
export interface ChatOptions<TAdapter extends AnyChatAdapter, TTools extends Record<string, AnyTool>> {
	adapter: TAdapter;
	messages: ModelMessage[];
	/**
	 * tools the model can call. names are the record keys; the same tool
	 * value can be registered under different names.
	 */
	tools?: TTools;
	temperature?: number;
	topP?: number;
	maxTokens?: number;
	/**
	 * controls when the agent loop stops. defaults to `maxIterations(5)`.
	 * when the strategy returns false before the model naturally finishes,
	 * a `finish` chunk with reason `length` is emitted.
	 *
	 * compose multiple strategies via `combineStrategies(...)`.
	 */
	agentLoopStrategy?: AgentLoopStrategy;
	signal?: AbortSignal;
	providerOptions?: TAdapter['~types']['providerOptions'];
}

/**
 * run an agentic chat turn against an adapter, executing any matching tools
 * and feeding their results back into the model until the model stops
 * requesting tools or `maxIterations` is reached.
 *
 * @returns an async iterable of {@link StreamChunk}s. tool-related chunks are
 * narrowed by `name` so `chunk.input` and `chunk.result` are typed against
 * the matching tool's schema.
 */
export const chat = <TAdapter extends AnyChatAdapter, const TTools extends Record<string, AnyTool> = {}>(
	options: ChatOptions<TAdapter, TTools>,
): AsyncIterable<StreamChunk<TTools>> => runChat(options);

/**
 * cast helper for chunks whose runtime shape is structurally a
 * `StreamChunk<TTools>` but whose `name` field carries a `string` instead
 * of the literal `keyof TTools` the type system requires. justified at the
 * boundary between the wire-typed adapter and the literal-typed public
 * stream.
 */
const passthrough = <TTools extends Record<string, AnyTool>>(
	chunk: object,
	// oxlint-disable-next-line typescript/no-unsafe-type-assertion
): StreamChunk<TTools> => chunk as StreamChunk<TTools>;

interface OpenTextPart {
	kind: 'text';
	id: string;
	text: string;
	providerMetadata: ProviderMetadata | undefined;
}

interface OpenReasoningPart {
	kind: 'reasoning';
	id: string;
	text: string;
	providerMetadata: ProviderMetadata | undefined;
}

interface OpenToolCallPart {
	kind: 'tool-call';
	id: string;
	name: string;
	args: string;
	providerMetadata: ProviderMetadata | undefined;
}

type OpenPart = OpenTextPart | OpenReasoningPart | OpenToolCallPart;

async function* runChat<TAdapter extends AnyChatAdapter, TTools extends Record<string, AnyTool>>(
	options: ChatOptions<TAdapter, TTools>,
): AsyncGenerator<StreamChunk<TTools>, void, void> {
	const {
		adapter,
		messages: initialMessages,
		tools: providedTools,
		agentLoopStrategy = maxIterationsStrategy(5),
		signal,
		...rest
	} = options;
	// oxlint-disable-next-line typescript/no-unsafe-type-assertion
	const tools: TTools = providedTools ?? ({} as TTools);

	const messages: ModelMessage[] = [...initialMessages];
	const wireTools = Object.entries(tools).map(([name, t]) => toWireTool(name, t));

	const state: AgentLoopState = {
		iterationCount: 0,
		lastFinishReason: undefined,
		toolCallsThisIteration: 0,
		totalToolCalls: 0,
		totalUsage: undefined,
	};

	// resume detection: if the conversation ends on an assistant turn whose
	// tool-call parts haven't been resolved by a following tool message,
	// process them in iteration 1 instead of calling the model again.
	let resumedToolCalls: ToolCallPart[] | undefined;
	{
		const tail = messages.at(-1);
		if (tail?.role === 'assistant') {
			const calls = tail.content.filter((p): p is ToolCallPart => p.type === 'tool-call');
			if (calls.length > 0) {
				resumedToolCalls = calls;
			}
		}
	}

	while (agentLoopStrategy(state)) {
		let toolCalls: ToolCallPart[];
		let finish: { reason: FinishReason; usage?: Usage; providerMetadata?: ProviderMetadata } | undefined;
		const validatedInputs = new Map<string, unknown>();

		if (resumedToolCalls) {
			toolCalls = resumedToolCalls;
			resumedToolCalls = undefined;
			for (const call of toolCalls) {
				const tool = tools[call.name];
				if (!tool) {
					// unknown tool — surfaced in the execute loop below
					continue;
				}
				// oxlint-disable-next-line eslint/no-await-in-loop
				const validated = await validateToolInput(tool, call.name, call.id, call.arguments);
				if (validated.kind === 'error') {
					yield passthrough<TTools>({ type: 'error', error: validated.error });
					return;
				}
				validatedInputs.set(call.id, validated.value);
			}
		} else {
			const open = new Map<string, OpenPart>();
			const partOrder: OpenPart[] = [];
			const assistantContent: AssistantContent = [];

			const closePart = (id: string): void => {
				const part = open.get(id);
				if (!part) {
					return;
				}
				open.delete(id);
				switch (part.kind) {
					case 'text': {
						assistantContent.push({
							type: 'text',
							text: part.text,
							providerMetadata: part.providerMetadata,
						});
						break;
					}
					case 'reasoning': {
						assistantContent.push({
							type: 'reasoning',
							text: part.text,
							providerMetadata: part.providerMetadata,
						});
						break;
					}
					case 'tool-call': {
						assistantContent.push({
							type: 'tool-call',
							id: part.id,
							name: part.name,
							arguments: part.args,
							providerMetadata: part.providerMetadata,
						});
						break;
					}
				}
			};

			// oxlint-disable-next-line eslint/no-await-in-loop
			for await (const chunk of adapter.chatStream({
				messages,
				tools: wireTools.length > 0 ? wireTools : undefined,
				signal,
				...rest,
			})) {
				switch (chunk.type) {
					case 'text-start': {
						const part: OpenTextPart = {
							kind: 'text',
							id: chunk.id,
							text: '',
							providerMetadata: chunk.providerMetadata,
						};
						open.set(chunk.id, part);
						partOrder.push(part);
						break;
					}
					case 'text-delta': {
						const part = open.get(chunk.id);
						if (part?.kind === 'text') {
							part.text += chunk.delta;
						}
						yield passthrough<TTools>({ type: 'text-delta', delta: chunk.delta });
						break;
					}
					case 'text-end': {
						const part = open.get(chunk.id);
						if (part?.kind === 'text' && chunk.providerMetadata) {
							part.providerMetadata = { ...part.providerMetadata, ...chunk.providerMetadata };
						}
						closePart(chunk.id);
						break;
					}
					case 'reasoning-start': {
						const part: OpenReasoningPart = {
							kind: 'reasoning',
							id: chunk.id,
							text: '',
							providerMetadata: chunk.providerMetadata,
						};
						open.set(chunk.id, part);
						partOrder.push(part);
						break;
					}
					case 'reasoning-delta': {
						const part = open.get(chunk.id);
						if (part?.kind === 'reasoning') {
							part.text += chunk.delta;
						}
						yield passthrough<TTools>({ type: 'reasoning-delta', delta: chunk.delta });
						break;
					}
					case 'reasoning-end': {
						const part = open.get(chunk.id);
						if (part?.kind === 'reasoning' && chunk.providerMetadata) {
							part.providerMetadata = { ...part.providerMetadata, ...chunk.providerMetadata };
						}
						closePart(chunk.id);
						break;
					}
					case 'redacted-reasoning': {
						assistantContent.push({ type: 'redacted-reasoning', providerMetadata: chunk.providerMetadata });
						break;
					}
					case 'tool-call-start': {
						const part: OpenToolCallPart = {
							kind: 'tool-call',
							id: chunk.id,
							name: chunk.name,
							args: '',
							providerMetadata: chunk.providerMetadata,
						};
						open.set(chunk.id, part);
						partOrder.push(part);
						yield passthrough<TTools>({ type: 'tool-call-start', name: chunk.name, id: chunk.id });
						break;
					}
					case 'tool-call-delta': {
						const part = open.get(chunk.id);
						if (part?.kind === 'tool-call') {
							part.args += chunk.argsDelta;
						}
						yield passthrough<TTools>({
							type: 'tool-call-delta',
							name: chunk.name,
							id: chunk.id,
							argsDelta: chunk.argsDelta,
						});
						break;
					}
					case 'tool-call-end': {
						const part = open.get(chunk.id);
						if (part?.kind === 'tool-call' && chunk.providerMetadata) {
							part.providerMetadata = { ...part.providerMetadata, ...chunk.providerMetadata };
						}
						closePart(chunk.id);
						if (part?.kind === 'tool-call') {
							const tool = tools[part.name];
							if (tool) {
								// oxlint-disable-next-line eslint/no-await-in-loop
								const validated = await validateToolInput(tool, part.name, part.id, part.args);
								if (validated.kind === 'error') {
									yield passthrough<TTools>({ type: 'error', error: validated.error });
									return;
								}
								validatedInputs.set(part.id, validated.value);
								yield passthrough<TTools>({
									type: 'tool-call-end',
									name: chunk.name,
									id: chunk.id,
									input: validated.value,
								});
							} else {
								// unknown tool — error surfaces in the execute loop below
								yield passthrough<TTools>({
									type: 'tool-call-end',
									name: chunk.name,
									id: chunk.id,
									input: undefined,
								});
							}
						}
						break;
					}
					case 'finish': {
						finish = { reason: chunk.reason, usage: chunk.usage, providerMetadata: chunk.providerMetadata };
						break;
					}
					case 'error': {
						yield passthrough<TTools>(chunk);
						return;
					}
				}
			}

			// close any parts the adapter forgot to terminate (defensive).
			for (const part of partOrder) {
				closePart(part.id);
			}

			toolCalls = assistantContent.filter((p): p is ToolCallPart => p.type === 'tool-call');

			const assistantMessage: AssistantMessage = {
				role: 'assistant',
				content: assistantContent,
				providerMetadata: finish?.providerMetadata,
			};
			messages.push(assistantMessage);
			yield passthrough<TTools>({ type: 'message', message: assistantMessage });

			const shouldExecuteTools =
				toolCalls.length > 0 && (finish?.reason === 'tool-calls' || finish === undefined);

			if (!shouldExecuteTools) {
				yield passthrough<TTools>({
					type: 'finish',
					reason: finish?.reason ?? 'stop',
					usage: finish?.usage,
				});
				return;
			}
		}

		// batch gate: any gated call without a decision suspends the entire
		// batch — siblings don't run speculatively, since the model emitted
		// them off the same context with no expectation of partial execution.
		// decisions live on `call.approval`; resume by mutating those fields
		// in the persisted assistant message and calling chat() again.
		const gated: ToolCallPart[] = [];
		for (const call of toolCalls) {
			const tool = tools[call.name];
			if (tool?.needsApproval && !call.approval) {
				gated.push(call);
			}
		}

		if (gated.length > 0) {
			for (const call of gated) {
				yield passthrough<TTools>({
					type: 'tool-approval-requested',
					name: call.name,
					id: call.id,
					input: validatedInputs.get(call.id),
				});
			}
			yield passthrough<TTools>({
				type: 'finish',
				reason: 'awaiting-approval',
				usage: finish?.usage,
			});
			return;
		}

		// tool calls execute sequentially: each tool's result is fed into the next
		// model call, and a failing tool aborts the rest of the loop. callers who
		// want parallelism should expose a single batch tool that fans out internally.
		const toolResults: ToolResultPart[] = [];
		for (const call of toolCalls) {
			const tool = tools[call.name];
			if (!tool) {
				yield passthrough<TTools>({
					type: 'error',
					error: new UnknownToolError(call.name, call.id),
				});
				return;
			}

			if (call.approval && !call.approval.approved) {
				const reason = 'user declined tool execution';
				yield passthrough<TTools>({
					type: 'tool-rejected',
					name: call.name,
					id: call.id,
					reason,
				});
				toolResults.push({ type: 'tool-result', toolCallId: call.id, output: reason, isError: true });
				continue;
			}

			// validatedInput populated above; re-validate as a fallback for adapters
			// that emit tool calls without a tool-call-end (none currently).
			let input: unknown;
			if (validatedInputs.has(call.id)) {
				input = validatedInputs.get(call.id);
			} else {
				// oxlint-disable-next-line eslint/no-await-in-loop
				const validated = await validateToolInput(tool, call.name, call.id, call.arguments);
				if (validated.kind === 'error') {
					yield passthrough<TTools>({ type: 'error', error: validated.error });
					return;
				}
				input = validated.value;
			}

			let result: unknown;
			try {
				// oxlint-disable-next-line eslint/no-await-in-loop
				result = await tool.execute(input, { toolCallId: call.id, signal });
			} catch (e) {
				yield passthrough<TTools>({ type: 'error', error: e });
				return;
			}

			if (tool.outputSchema) {
				// oxlint-disable-next-line eslint/no-await-in-loop
				const validated = await validateToolOutput(tool, call.name, call.id, result);
				if (validated.kind === 'error') {
					yield passthrough<TTools>({ type: 'error', error: validated.error });
					return;
				}
				result = validated.value;
			}

			yield passthrough<TTools>({
				type: 'tool-result',
				name: call.name,
				id: call.id,
				result,
			});

			toolResults.push({
				type: 'tool-result',
				toolCallId: call.id,
				output: typeof result === 'string' ? result : JSON.stringify(result),
			});
		}

		const toolMessage: ToolMessage = { role: 'tool', content: toolResults };
		messages.push(toolMessage);
		yield passthrough<TTools>({ type: 'message', message: toolMessage });

		// update strategy state after a tools-executed iteration
		state.iterationCount++;
		state.lastFinishReason = finish?.reason;
		state.toolCallsThisIteration = toolCalls.length;
		state.totalToolCalls += toolCalls.length;
		if (finish?.usage) {
			state.totalUsage = sumUsage(state.totalUsage, finish.usage);
		}
	}

	// strategy declined to continue while the model still wanted to call tools
	yield passthrough<TTools>({ type: 'finish', reason: 'length', usage: state.totalUsage });
}

const sumUsage = (a: Usage | undefined, b: Usage): Usage => {
	const out: Usage = {
		inputTokens: (a?.inputTokens ?? 0) + b.inputTokens,
		outputTokens: (a?.outputTokens ?? 0) + b.outputTokens,
	};
	const cacheCreate = (a?.cacheCreationInputTokens ?? 0) + (b.cacheCreationInputTokens ?? 0);
	if (
		cacheCreate > 0 ||
		a?.cacheCreationInputTokens !== undefined ||
		b.cacheCreationInputTokens !== undefined
	) {
		out.cacheCreationInputTokens = cacheCreate;
	}
	const cacheRead = (a?.cacheReadInputTokens ?? 0) + (b.cacheReadInputTokens ?? 0);
	if (cacheRead > 0 || a?.cacheReadInputTokens !== undefined || b.cacheReadInputTokens !== undefined) {
		out.cacheReadInputTokens = cacheRead;
	}
	const reasoning = (a?.reasoningTokens ?? 0) + (b.reasoningTokens ?? 0);
	if (reasoning > 0 || a?.reasoningTokens !== undefined || b.reasoningTokens !== undefined) {
		out.reasoningTokens = reasoning;
	}
	return out;
};

type ValidateResult = { kind: 'ok'; value: unknown } | { kind: 'error'; error: Error };

const validateToolInput = async (
	tool: AnyTool,
	name: string,
	id: string,
	rawArgs: string,
): Promise<ValidateResult> => {
	let parsed: unknown;
	try {
		parsed = rawArgs.length > 0 ? JSON.parse(rawArgs) : {};
	} catch (e) {
		return {
			kind: 'error',
			error: new ToolArgumentsParseError(name, id, rawArgs, e),
		};
	}

	const result = await tool.inputSchema['~standard'].validate(parsed);
	if (result.issues) {
		return {
			kind: 'error',
			error: new ToolInputValidationError(name, id, rawArgs, result.issues),
		};
	}
	return { kind: 'ok', value: result.value };
};

const validateToolOutput = async (
	tool: AnyTool,
	name: string,
	id: string,
	result: unknown,
): Promise<ValidateResult> => {
	const schema = tool.outputSchema;
	if (!schema) {
		return { kind: 'ok', value: result };
	}

	const validated = await schema['~standard'].validate(result);
	if (validated.issues) {
		return {
			kind: 'error',
			error: new ToolOutputValidationError(name, id, result, validated.issues),
		};
	}
	return { kind: 'ok', value: validated.value };
};

// jsonSchema generation can be non-trivial (e.g. zod's toJSONSchema). cache
// per (tool, name) so each chat() call doesn't re-derive the same wire shape.
const wireToolCache = new WeakMap<AnyTool, Map<string, WireTool>>();

const toWireTool = (name: string, tool: AnyTool): WireTool => {
	let byName = wireToolCache.get(tool);
	if (!byName) {
		byName = new Map();
		wireToolCache.set(tool, byName);
	}
	const cached = byName.get(name);
	if (cached) {
		return cached;
	}
	const schema: ToolSchema = tool.inputSchema;
	const wire: WireTool = {
		name,
		description: tool.description,
		jsonSchema: schema['~standard'].jsonSchema.input({ target: 'draft-07' }),
	};
	byName.set(name, wire);
	return wire;
};
