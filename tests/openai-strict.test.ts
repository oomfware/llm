import assert from 'node:assert/strict';
import { test } from 'node:test';

import { z } from 'zod';

import { generateObject } from '../src/generate-object.ts';
import { makeOpenAIStrictCompatible } from '../src/internal/openai-strict.ts';
import { user } from '../src/messages.ts';
import { openai } from '../src/providers/openai.ts';

test('makeOpenAIStrictCompatible: forces additionalProperties: false on every object', () => {
	const out = makeOpenAIStrictCompatible({
		type: 'object',
		properties: {
			a: { type: 'string' },
			nested: {
				type: 'object',
				properties: { b: { type: 'number' } },
				required: ['b'],
			},
		},
		required: ['a', 'nested'],
	});

	assert.equal(out.additionalProperties, false);
	const nested = (out.properties as Record<string, Record<string, unknown>>).nested;
	assert.equal(nested?.additionalProperties, false);
});

test('makeOpenAIStrictCompatible: optional fields become nullable + listed in required', () => {
	const out = makeOpenAIStrictCompatible({
		type: 'object',
		properties: {
			required_field: { type: 'string' },
			optional_field: { type: 'string' },
		},
		required: ['required_field'],
	});

	assert.deepEqual(out.required, ['required_field', 'optional_field']);
	const props = out.properties as Record<string, Record<string, unknown>>;
	assert.equal(props.required_field?.type, 'string');
	assert.deepEqual(props.optional_field?.type, ['string', 'null']);
});

test('makeOpenAIStrictCompatible: recurses into items/anyOf/$defs', () => {
	const out = makeOpenAIStrictCompatible({
		type: 'object',
		properties: {
			arr: {
				type: 'array',
				items: { type: 'object', properties: { x: { type: 'number' } } },
			},
			either: {
				anyOf: [
					{ type: 'object', properties: { y: { type: 'number' } } },
					{ type: 'object', properties: { z: { type: 'string' } } },
				],
			},
		},
		required: ['arr', 'either'],
	});

	const props = out.properties as Record<string, Record<string, unknown>>;
	const arrItems = props.arr?.items as Record<string, unknown>;
	assert.equal(arrItems.additionalProperties, false);

	const eitherAnyOf = props.either?.anyOf as Array<Record<string, unknown>>;
	assert.equal(eitherAnyOf[0]?.additionalProperties, false);
	assert.equal(eitherAnyOf[1]?.additionalProperties, false);
});

test('openai structuredOutput preprocesses schema and strips nulls from response', async () => {
	let capturedSchema: any;
	const fakeFetch = async (_url: string | URL | Request, init?: RequestInit) => {
		const body = JSON.parse(init?.body as string);
		if (body.text?.format) {
			capturedSchema = body.text.format.schema;
			// model returns nullable optional field as `null`; we should strip it
			return new Response(
				JSON.stringify({
					output_text: JSON.stringify({ name: 'cat', alias: null }),
				}),
				{ status: 200, headers: { 'content-type': 'application/json' } },
			);
		}
		// the streaming chat call that runs before structuredOutput — finish quickly
		const enc = new TextEncoder();
		const body_ = new ReadableStream<Uint8Array>({
			start(controller) {
				controller.enqueue(
					enc.encode(
						`data: ${JSON.stringify({ choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] })}\n\n`,
					),
				);
				controller.enqueue(enc.encode(`data: [DONE]\n\n`));
				controller.close();
			},
		});
		return new Response(body_, {
			status: 200,
			headers: { 'content-type': 'text/event-stream' },
		});
	};

	const adapter = openai('gpt-4o', { apiKey: 'test', fetch: fakeFetch as typeof fetch });

	const { object } = await generateObject({
		adapter,
		messages: [user('name something')],
		schema: z.object({ name: z.string(), alias: z.string().optional() }),
	});

	// schema sent to openai had additionalProperties: false and `alias` made nullable
	assert.equal(capturedSchema.additionalProperties, false);
	assert.deepEqual(capturedSchema.required, ['name', 'alias']);
	assert.deepEqual(capturedSchema.properties.alias.type, ['string', 'null']);

	// null `alias` was stripped on the way out
	assert.equal(object.name, 'cat');
	assert.equal(object.alias, undefined);
});
