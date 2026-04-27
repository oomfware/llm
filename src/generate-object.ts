import type { StandardJSONSchemaV1, StandardSchemaV1 } from '@standard-schema/spec';

import type { AnyChatAdapter } from './adapter.ts';
import type { AgentLoopStrategy } from './agent-loop-strategies.ts';
import { StructuredOutputValidationError } from './errors.ts';
import { generate } from './generate.ts';
import type { AnyTool } from './tool.ts';
import type { AssistantMessage, FinishReason, ModelMessage, Usage } from './types.ts';

/**
 * the schema a structured-output call constrains its result to. must be both
 * Standard Schema (for validation) and Standard JSON Schema (for the wire).
 */
export type ObjectSchema<T> = StandardSchemaV1<unknown, T> & StandardJSONSchemaV1<unknown, T>;

export interface GenerateObjectOptions<
	TAdapter extends AnyChatAdapter,
	TSchema extends StandardSchemaV1 & StandardJSONSchemaV1,
	TTools extends Record<string, AnyTool>,
> {
	adapter: TAdapter;
	messages: ModelMessage[];
	schema: TSchema;
	/** optional schema label some providers expose (e.g. OpenAI's `json_schema.name`). */
	schemaName?: string;
	tools?: TTools;
	temperature?: number;
	topP?: number;
	maxTokens?: number;
	agentLoopStrategy?: AgentLoopStrategy;
	signal?: AbortSignal;
	providerOptions?: TAdapter['~types']['providerOptions'];
}

export interface GenerateObjectResult<T> {
	object: T;
	/** raw text the structured output was parsed from. */
	rawText: string;
	/** finish reason of the final structured-output call. */
	finishReason: FinishReason;
	/** combined usage from the agent loop and the final structured-output call. */
	usage: Usage | undefined;
	/** the conversation up to and including the final structured-output assistant turn. */
	messages: ModelMessage[];
}

/**
 * run an agentic chat to convergence, then make a single non-streaming model
 * call constrained to {@link ObjectSchema} and return the validated object.
 *
 * @throws when the provider returns malformed JSON or when validation against
 * the schema fails.
 */
export const generateObject = async <
	TAdapter extends AnyChatAdapter,
	TSchema extends StandardSchemaV1 & StandardJSONSchemaV1,
	const TTools extends Record<string, AnyTool> = {},
>(
	options: GenerateObjectOptions<TAdapter, TSchema, TTools>,
): Promise<GenerateObjectResult<StandardSchemaV1.InferOutput<TSchema>>> => {
	const { adapter, schema, schemaName, ...rest } = options;

	// step 1: run the agent loop to convergence (tool calls executed, conversation finalized)
	const loop = await generate({ ...rest, adapter });

	// step 2: extract JSON Schema and ask the provider for a structured response
	const jsonSchema = schema['~standard'].jsonSchema.output({ target: 'draft-07' });

	const result = await adapter.structuredOutput({
		messages: loop.messages,
		jsonSchema,
		name: schemaName,
		temperature: options.temperature,
		topP: options.topP,
		maxTokens: options.maxTokens,
		signal: options.signal,
		providerOptions: options.providerOptions,
	});

	// step 3: validate against the Standard Schema
	const validated = await schema['~standard'].validate(result.data);
	if (validated.issues) {
		throw new StructuredOutputValidationError(result.rawText, result.data, validated.issues);
	}

	const finalAssistant: AssistantMessage = {
		role: 'assistant',
		content: [{ type: 'text', text: result.rawText }],
	};
	const messages: ModelMessage[] = [...loop.messages, finalAssistant];

	return {
		// oxlint-disable-next-line typescript/no-unsafe-type-assertion
		object: validated.value as StandardSchemaV1.InferOutput<TSchema>,
		rawText: result.rawText,
		finishReason: loop.finishReason,
		usage: loop.usage,
		messages,
	};
};
