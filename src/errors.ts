import type { StandardSchemaV1 } from '@standard-schema/spec';

/**
 * base class for every error this library throws (or yields as the `error`
 * field of an `error` chunk). subclass `instanceof` checks let callers
 * branch on failure mode without parsing message strings.
 */
export class AIError extends Error {
	override name = 'AIError';
}

/**
 * the model called a tool name that wasn't registered for the current turn.
 */
export class UnknownToolError extends AIError {
	override name = 'UnknownToolError';
	readonly toolName: string;
	readonly toolCallId: string;

	constructor(toolName: string, toolCallId: string) {
		super(`unknown tool: ${toolName}`);
		this.toolName = toolName;
		this.toolCallId = toolCallId;
	}
}

/**
 * the model's tool-call arguments string wasn't parseable as json. the
 * underlying parse error is preserved on `cause`.
 */
export class ToolArgumentsParseError extends AIError {
	override name = 'ToolArgumentsParseError';
	readonly toolName: string;
	readonly toolCallId: string;
	readonly rawArguments: string;

	constructor(toolName: string, toolCallId: string, rawArguments: string, cause: unknown) {
		super('failed to parse tool arguments as json', { cause });
		this.toolName = toolName;
		this.toolCallId = toolCallId;
		this.rawArguments = rawArguments;
	}
}

/**
 * the model's tool-call arguments parsed as json but failed the tool's
 * `inputSchema`. `issues` is the standard-schema issue array.
 */
export class ToolInputValidationError extends AIError {
	override name = 'ToolInputValidationError';
	readonly toolName: string;
	readonly toolCallId: string;
	readonly rawArguments: string;
	readonly issues: readonly StandardSchemaV1.Issue[];

	constructor(
		toolName: string,
		toolCallId: string,
		rawArguments: string,
		issues: readonly StandardSchemaV1.Issue[],
	) {
		super(`tool input validation failed: ${issues.map((i) => i.message).join('; ')}`);
		this.toolName = toolName;
		this.toolCallId = toolCallId;
		this.rawArguments = rawArguments;
		this.issues = issues;
	}
}

/**
 * the value returned by a tool's `execute` failed its `outputSchema`. the
 * raw return value is preserved on `rawResult` for inspection.
 */
export class ToolOutputValidationError extends AIError {
	override name = 'ToolOutputValidationError';
	readonly toolName: string;
	readonly toolCallId: string;
	readonly rawResult: unknown;
	readonly issues: readonly StandardSchemaV1.Issue[];

	constructor(
		toolName: string,
		toolCallId: string,
		rawResult: unknown,
		issues: readonly StandardSchemaV1.Issue[],
	) {
		super(`tool output validation failed: ${issues.map((i) => i.message).join('; ')}`);
		this.toolName = toolName;
		this.toolCallId = toolCallId;
		this.rawResult = rawResult;
		this.issues = issues;
	}
}

/**
 * `generateObject()`'s final structured-output call returned data that
 * didn't match the supplied schema. carries the raw response text and the
 * parsed (but unvalidated) data alongside the schema issues.
 */
export class StructuredOutputValidationError extends AIError {
	override name = 'StructuredOutputValidationError';
	readonly rawText: string;
	readonly rawData: unknown;
	readonly issues: readonly StandardSchemaV1.Issue[];

	constructor(rawText: string, rawData: unknown, issues: readonly StandardSchemaV1.Issue[]) {
		super(`structured output failed schema validation: ${issues.map((i) => i.message).join('; ')}`);
		this.rawText = rawText;
		this.rawData = rawData;
		this.issues = issues;
	}
}
