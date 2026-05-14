import type { ToolSchema } from './tool.ts';

/**
 * wrap a plain JSON Schema as a Standard JSON Schema-compatible value, so it
 * can be passed to `tool({ inputSchema })` or `generateObject({ schema })`
 * without requiring zod/valibot/arktype.
 *
 * the type parameter `T` declares the inferred shape — it isn't checked
 * against the schema, so it's the caller's job to keep them in sync.
 *
 * by default no runtime validation is performed (the parsed JSON is passed
 * through). pass `parse` for a custom validator that throws on invalid input
 * — its return value is what `execute` receives.
 *
 * @example
 * ```ts
 * import { jsonSchema, tool } from '@oomfware/llm';
 *
 * const getWeather = tool({
 *   inputSchema: jsonSchema<{ city: string }>({
 *     type: 'object',
 *     properties: { city: { type: 'string' } },
 *     required: ['city'],
 *   }),
 *   execute: ({ city }) => `weather in ${city}`,
 * });
 * ```
 */
export const jsonSchema = <T = unknown>(
	schema: Record<string, unknown>,
	parse?: (raw: unknown) => T,
): ToolSchema<unknown, T> => {
	const validate = parse
		? (value: unknown) => {
				try {
					return { value: parse(value) };
				} catch (e) {
					const message = e instanceof Error ? e.message : String(e);
					return { issues: [{ message }] };
				}
			}
		: // oxlint-disable-next-line typescript/no-unsafe-type-assertion
			(value: unknown) => ({ value: value as T });

	return {
		'~standard': {
			version: 1,
			vendor: 'plain-json-schema',
			validate,
			jsonSchema: {
				input: () => schema,
				output: () => schema,
			},
		},
	};
};
