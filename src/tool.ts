import type { StandardJSONSchemaV1, StandardSchemaV1 } from '@standard-schema/spec';

/**
 * a schema must be both Standard Schema (for validation) and Standard JSON
 * Schema (for converting to the wire format the model sees). modern schema
 * libs that support standardschema.dev/json-schema (zod v4.2+, arktype v2.1.28+,
 * valibot v1.2+) satisfy this directly.
 */
export type ToolSchema<TInput = unknown, TOutput = TInput> = StandardSchemaV1<TInput, TOutput> &
	StandardJSONSchemaV1<TInput, TOutput>;

export interface ToolContext {
	toolCallId: string;
	signal?: AbortSignal;
}

/**
 * a tool the model can call. names are assigned at registration site
 * (`chat({ tools: { my_tool: ... } })`), not on the tool itself, so the
 * same tool can be reused under different keys.
 */
export interface Tool<TInput = unknown, TOutput = unknown> {
	description?: string;
	inputSchema: ToolSchema<unknown, TInput>;
	/**
	 * optional output schema. when set, the value returned by `execute` is
	 * validated against it before being yielded as a `tool-result` chunk;
	 * a validation failure aborts the agent loop with an `error` chunk.
	 *
	 * the schema must accept whatever `execute` may return — including any
	 * intentional `null`/`undefined` cases (use `.nullable()` / `.optional()`).
	 */
	outputSchema?: ToolSchema<unknown, TOutput>;
	/**
	 * when true, the agent loop suspends before executing this tool and
	 * yields a `tool-approval-requested` chunk for each gated call in the
	 * batch, finishing with reason `awaiting-approval`. resume by setting
	 * `approval` on each pending tool call in the persisted assistant
	 * message and calling `chat()` again with the same `messages` array.
	 *
	 * any one gated call in a batch suspends *all* sibling calls — none
	 * execute until every gated call in the batch has a decision.
	 */
	needsApproval?: boolean;
	execute: (input: TInput, ctx: ToolContext) => TOutput | Promise<TOutput>;
}

export type AnyTool = Tool<any, any>;

// #region tool() factory overloads

/**
 * with both `inputSchema` and `outputSchema`: `execute`'s return is constrained
 * to the output schema's inferred type, and the tool's `TOutput` flows from there.
 */
export function tool<
	TIn extends StandardSchemaV1 & StandardJSONSchemaV1,
	TOut extends StandardSchemaV1 & StandardJSONSchemaV1,
>(config: {
	description?: string;
	inputSchema: TIn;
	outputSchema: TOut;
	needsApproval?: boolean;
	execute: (
		input: StandardSchemaV1.InferOutput<TIn>,
		ctx: ToolContext,
	) => StandardSchemaV1.InferOutput<TOut> | Promise<StandardSchemaV1.InferOutput<TOut>>;
	// oxlint-disable-next-line typescript/no-unnecessary-type-arguments
}): Tool<StandardSchemaV1.InferOutput<TIn>, StandardSchemaV1.InferOutput<TOut>>;

/**
 * with only `inputSchema`: `TOutput` is inferred from `execute`'s return.
 */
export function tool<TIn extends StandardSchemaV1 & StandardJSONSchemaV1, TOut>(config: {
	description?: string;
	inputSchema: TIn;
	needsApproval?: boolean;
	execute: (input: StandardSchemaV1.InferOutput<TIn>, ctx: ToolContext) => TOut | Promise<TOut>;
}): Tool<StandardSchemaV1.InferOutput<TIn>, Awaited<TOut>>;

export function tool(config: {
	description?: string;
	inputSchema: ToolSchema;
	outputSchema?: ToolSchema;
	needsApproval?: boolean;
	execute: (input: any, ctx: ToolContext) => unknown;
}): Tool {
	return config as Tool;
}

// #endregion

/** infer the (validated) input type a tool's `execute` receives. */
export type InferToolInput<T> = T extends Tool<infer I, any> ? I : never;

/** infer the output type a tool's `execute` returns. */
export type InferToolOutput<T> = T extends Tool<any, infer O> ? O : never;
