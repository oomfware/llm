// rewriting a schema is non-trivial for large shapes; cache per input reference so repeat structured-output
// calls against the same schema object don't re-walk the tree.
const strictCache = new WeakMap<Record<string, unknown>, Record<string, unknown>>();

/**
 * rewrite a JSON Schema for openai's `strict: true` mode: - every object gets `additionalProperties: false` -
 * every property must appear in `required` (originally-optional fields are made nullable to preserve their
 * meaning)
 *
 * recurses into properties, items, anyOf/oneOf/allOf, $defs/definitions.
 */
export const makeOpenAIStrictCompatible = (schema: Record<string, unknown>): Record<string, unknown> => {
	const cached = strictCache.get(schema);
	if (cached) {
		return cached;
	}
	const rewritten = rewriteSchema(schema);
	strictCache.set(schema, rewritten);
	return rewritten;
};

const rewriteSchema = (schema: unknown): Record<string, unknown> => {
	if (typeof schema !== 'object' || schema === null) {
		// oxlint-disable-next-line typescript/no-unsafe-type-assertion
		return schema as Record<string, unknown>;
	}
	// oxlint-disable-next-line typescript/no-unsafe-type-assertion
	const out: Record<string, unknown> = { ...(schema as Record<string, unknown>) };

	// oxlint-disable-next-line typescript/no-unsafe-type-assertion
	const properties = out.properties as Record<string, unknown> | undefined;
	if (properties) {
		// oxlint-disable-next-line typescript/no-unsafe-type-assertion
		const originalRequired = new Set((out.required as string[] | undefined) ?? []);
		const allKeys = Object.keys(properties);
		const rewrittenProps: Record<string, unknown> = {};
		for (const key of allKeys) {
			const rewritten = rewriteSchema(properties[key]);
			rewrittenProps[key] = originalRequired.has(key) ? rewritten : makeNullable(rewritten);
		}
		out.properties = rewrittenProps;
		out.required = allKeys;
		out.additionalProperties = false;
	}

	if (out.items !== undefined) {
		out.items = Array.isArray(out.items) ? out.items.map(rewriteSchema) : rewriteSchema(out.items);
	}

	for (const key of ['anyOf', 'oneOf', 'allOf'] as const) {
		const arr = out[key];
		if (Array.isArray(arr)) {
			out[key] = arr.map(rewriteSchema);
		}
	}

	for (const key of ['$defs', 'definitions'] as const) {
		const defs = out[key];
		if (defs && typeof defs === 'object') {
			const rewritten: Record<string, unknown> = {};
			for (const [k, v] of Object.entries(defs)) {
				rewritten[k] = rewriteSchema(v);
			}
			out[key] = rewritten;
		}
	}

	return out;
};

const makeNullable = (schema: Record<string, unknown>): Record<string, unknown> => {
	const t = schema.type;
	if (typeof t === 'string') {
		return { ...schema, type: [t, 'null'] };
	}
	if (Array.isArray(t)) {
		return t.includes('null') ? schema : { ...schema, type: [...t, 'null'] };
	}
	if (Array.isArray(schema.anyOf)) {
		return { ...schema, anyOf: [...schema.anyOf, { type: 'null' }] };
	}
	return { anyOf: [schema, { type: 'null' }] };
};

/**
 * strip top-level `null` values that originated from optional fields we made nullable for strict mode.
 * recursively descends into nested objects/arrays.
 */
export const stripNulls = (value: unknown): unknown => {
	if (value === null) {
		return undefined;
	}
	if (Array.isArray(value)) {
		return value.map(stripNulls);
	}
	if (typeof value === 'object') {
		const out: Record<string, unknown> = {};
		for (const [k, v] of Object.entries(value)) {
			const transformed = stripNulls(v);
			if (transformed !== undefined) {
				out[k] = transformed;
			}
		}
		return out;
	}
	return value;
};
