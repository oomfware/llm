import assert from 'node:assert/strict';
import { test } from 'node:test';

import { z } from 'zod';

import { StructuredOutputValidationError } from '../src/errors.ts';
import { generateObject } from '../src/generate-object.ts';
import { user } from '../src/messages.ts';
import { dummy } from '../src/providers/dummy.ts';
import { tool } from '../src/tool.ts';

test('returns a typed object validated against the schema', async () => {
	const adapter = dummy({
		responses: [{ text: 'thinking...' }],
		structuredOutputs: [{ altText: 'a cat sleeping', clarifyingQuestions: ['what breed?'] }],
	});

	const result = await generateObject({
		adapter,
		messages: [user('describe the image')],
		schema: z.object({
			altText: z.string(),
			clarifyingQuestions: z.array(z.string()),
		}),
	});

	assert.deepEqual(result.object, {
		altText: 'a cat sleeping',
		clarifyingQuestions: ['what breed?'],
	});

	// the structured output call receives the conversation including the prior assistant turn
	assert.equal(adapter.structuredCalls.length, 1);
	const lastMessage = adapter.structuredCalls[0]?.messages.at(-1);
	assert.equal(lastMessage?.role, 'assistant');
});

test('runs the agent loop with tools before structured output', async () => {
	const lookupSize = tool({
		inputSchema: z.object({ url: z.string() }),
		execute: ({ url }) => ({ url, width: 800, height: 600 }),
	});

	const adapter = dummy({
		responses: [
			{
				toolCalls: [{ id: 'c_1', name: 'lookup_size', arguments: { url: 'cat.png' } }],
			},
			{ text: 'ok, I have the dimensions.' },
		],
		structuredOutputs: [{ width: 800, height: 600 }],
	});

	const result = await generateObject({
		adapter,
		messages: [user('how big is cat.png?')],
		tools: { lookup_size: lookupSize },
		schema: z.object({ width: z.number(), height: z.number() }),
	});

	assert.deepEqual(result.object, { width: 800, height: 600 });
	// two streaming calls (tool loop) + one structured output call
	assert.equal(adapter.calls.length, 2);
	assert.equal(adapter.structuredCalls.length, 1);
});

test('throws when validation fails', async () => {
	const adapter = dummy({
		responses: [{ text: 'thinking...' }],
		structuredOutputs: [{ altText: 'cat' }],
	});

	await assert.rejects(
		() =>
			generateObject({
				adapter,
				messages: [user('go')],
				schema: z.object({
					altText: z.string(),
					clarifyingQuestions: z.array(z.string()),
				}),
			}),
		(err) => err instanceof StructuredOutputValidationError,
	);
});
