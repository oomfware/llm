import assert from 'node:assert/strict';
import { test } from 'node:test';

import { generate } from '../src/generate.ts';
import { jsonSchema } from '../src/json-schema.ts';
import { user } from '../src/messages.ts';
import { dummy } from '../src/providers/dummy.ts';
import { tool } from '../src/tool.ts';

test('jsonSchema() wraps a plain schema for use with tool()', async () => {
	const echo = tool({
		inputSchema: jsonSchema<{ msg: string }>({
			type: 'object',
			properties: { msg: { type: 'string' } },
			required: ['msg'],
		}),
		execute: ({ msg }) => msg.toUpperCase(),
	});

	const adapter = dummy({
		responses: [{ toolCalls: [{ id: 'c', name: 'echo', arguments: { msg: 'hi' } }] }, { text: 'done' }],
	});

	const result = await generate({
		adapter,
		messages: [user('go')],
		tools: { echo },
	});

	assert.equal(result.toolResults.length, 1);
	assert.equal(result.toolResults[0]?.result, 'HI');
});

test('jsonSchema() with parse runs validation', async () => {
	const strict = tool({
		inputSchema: jsonSchema<{ count: number }>(
			{ type: 'object', properties: { count: { type: 'number' } }, required: ['count'] },
			(raw) => {
				const r = raw as { count: unknown };
				if (typeof r.count !== 'number' || r.count < 0) {
					throw new Error('count must be non-negative');
				}
				return { count: r.count };
			},
		),
		execute: ({ count }) => count * 2,
	});

	const adapter = dummy({
		responses: [{ toolCalls: [{ id: 'c', name: 'strict', arguments: { count: -1 } }] }],
	});

	await assert.rejects(
		() =>
			generate({
				adapter,
				messages: [user('go')],
				tools: { strict },
			}),
		/count must be non-negative/,
	);
});

test('jsonSchema() without parse passes raw value through', async () => {
	const schema = jsonSchema<{ x: number }>({
		type: 'object',
		properties: { x: { type: 'number' } },
	});

	// validate is a no-op identity
	const result = await schema['~standard'].validate({ x: 42 });
	assert.deepEqual(result, { value: { x: 42 } });

	// jsonSchema accessor returns the raw object on both sides
	const inputJson = schema['~standard'].jsonSchema.input({ target: 'draft-07' });
	assert.deepEqual(inputJson, {
		type: 'object',
		properties: { x: { type: 'number' } },
	});
});
