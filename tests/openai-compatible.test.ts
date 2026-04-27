import assert from 'node:assert/strict';
import { test } from 'node:test';

import { z } from 'zod';

import { chat } from '../src/chat.ts';
import { generateObject } from '../src/generate-object.ts';
import { user } from '../src/messages.ts';
import { openaiCompatible } from '../src/providers/openai-compatible.ts';
import { tool } from '../src/tool.ts';

import { collect, sseResponse } from './_helpers.ts';

test('openaiCompatible parses a Chat Completions text stream and normalizes finish reason', async () => {
	const events = [
		JSON.stringify({ choices: [{ index: 0, delta: { content: 'hello ' } }] }),
		JSON.stringify({ choices: [{ index: 0, delta: { content: 'world' } }] }),
		JSON.stringify({ choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] }),
		JSON.stringify({ choices: [], usage: { prompt_tokens: 10, completion_tokens: 2 } }),
		'[DONE]',
	];

	const fakeFetch = async () => sseResponse(events);

	const adapter = openaiCompatible('test-model', {
		name: 'compat',
		baseUrl: 'https://compat.example/v1',
		apiKey: 'test',
		fetch: fakeFetch as typeof fetch,
	});

	const chunks = await collect(chat({ adapter, messages: [user('hi')] }));

	const text = chunks
		.filter((c) => c.type === 'text-delta')
		.map((c) => c.delta)
		.join('');
	assert.equal(text, 'hello world');

	const finish = chunks.at(-1);
	assert.equal(finish?.type, 'finish');
	if (finish?.type === 'finish') {
		assert.equal(finish.reason, 'stop');
		assert.equal(finish.usage?.inputTokens, 10);
		assert.equal(finish.usage?.outputTokens, 2);
	}
});

test('openaiCompatible parses Chat Completions tool-call deltas and feeds tool results back', async () => {
	const getWeather = tool({
		description: 'get the weather',
		inputSchema: z.object({ city: z.string() }),
		execute: ({ city }) => `sunny in ${city}`,
	});

	let callCount = 0;
	const fakeFetch = async (_url: string | URL | Request, init?: RequestInit) => {
		callCount++;
		if (callCount === 1) {
			return sseResponse([
				JSON.stringify({
					choices: [
						{
							index: 0,
							delta: {
								tool_calls: [
									{
										index: 0,
										id: 'call_1',
										type: 'function',
										function: { name: 'get_weather', arguments: '' },
									},
								],
							},
						},
					],
				}),
				JSON.stringify({
					choices: [
						{
							index: 0,
							delta: { tool_calls: [{ index: 0, function: { arguments: '{"city":' } }] },
						},
					],
				}),
				JSON.stringify({
					choices: [
						{
							index: 0,
							delta: { tool_calls: [{ index: 0, function: { arguments: '"paris"}' } }] },
						},
					],
				}),
				JSON.stringify({ choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }] }),
				'[DONE]',
			]);
		}

		const body = JSON.parse(init?.body as string) as {
			messages: { role: string; tool_call_id?: string; content?: string }[];
		};
		const lastMessage = body.messages.at(-1)!;
		assert.equal(lastMessage.role, 'tool');
		assert.equal(lastMessage.tool_call_id, 'call_1');
		assert.equal(lastMessage.content, 'sunny in paris');

		return sseResponse([
			JSON.stringify({ choices: [{ index: 0, delta: { content: 'reported.' } }] }),
			JSON.stringify({ choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] }),
			'[DONE]',
		]);
	};

	const adapter = openaiCompatible('test-model', {
		name: 'compat',
		baseUrl: 'https://compat.example/v1',
		apiKey: 'test',
		fetch: fakeFetch as typeof fetch,
	});

	const chunks = await collect(
		chat({
			adapter,
			messages: [user('weather?')],
			tools: { get_weather: getWeather },
		}),
	);

	assert.equal(callCount, 2);

	const toolStart = chunks.find((c) => c.type === 'tool-call-start');
	assert.equal(toolStart?.type, 'tool-call-start');
	if (toolStart?.type === 'tool-call-start') {
		assert.equal(toolStart.name, 'get_weather');
		assert.equal(toolStart.id, 'call_1');
	}

	const toolEnd = chunks.find((c) => c.type === 'tool-call-end');
	if (toolEnd?.type === 'tool-call-end' && toolEnd.name === 'get_weather') {
		assert.deepEqual(toolEnd.input, { city: 'paris' });
	}

	const toolResult = chunks.find((c) => c.type === 'tool-result');
	if (toolResult?.type === 'tool-result' && toolResult.name === 'get_weather') {
		assert.equal(toolResult.result, 'sunny in paris');
	}

	const finalText = chunks
		.filter((c) => c.type === 'text-delta')
		.map((c) => c.delta)
		.join('');
	assert.equal(finalText, 'reported.');
});

test('openaiCompatible forwards Chat Completions provider options and extension fields', async () => {
	let captured: Record<string, unknown> | undefined;
	const fakeFetch = async (_url: string | URL | Request, init?: RequestInit) => {
		captured = JSON.parse(init?.body as string);
		return sseResponse([
			JSON.stringify({ choices: [{ index: 0, delta: { content: 'ok' } }] }),
			JSON.stringify({ choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] }),
			'[DONE]',
		]);
	};

	const adapter = openaiCompatible('test-model', {
		name: 'compat',
		baseUrl: 'https://compat.example/v1',
		apiKey: 'test',
		fetch: fakeFetch as typeof fetch,
		extendBody: (opts) => ({ copied_seed: opts.seed }),
	});

	const stream = chat({
		adapter,
		messages: [user('hi')],
		providerOptions: {
			reasoningEffort: 'high',
			parallelToolCalls: false,
			seed: 123,
		},
	});
	for await (const _ of stream) {
		// drain
	}

	assert.equal(captured?.reasoning_effort, 'high');
	assert.equal(captured?.parallel_tool_calls, false);
	assert.equal(captured?.seed, 123);
	assert.equal(captured?.copied_seed, 123);
});

test('openaiCompatible surfaces a non-2xx response as an error chunk', async () => {
	const fakeFetch = async () => new Response('bad request', { status: 400, statusText: 'Bad Request' });

	const adapter = openaiCompatible('test-model', {
		name: 'compat',
		baseUrl: 'https://compat.example/v1',
		apiKey: 'test',
		fetch: fakeFetch as typeof fetch,
	});

	const chunks = await collect(chat({ adapter, messages: [user('hi')] }));

	const error = chunks.find((c) => c.type === 'error');
	assert.ok(error, 'expected an error chunk');
	if (error?.type === 'error') {
		assert.match((error.error as Error).message, /compat: 400/);
	}
});

test('openaiCompatible sends Chat Completions tool schemas on the wire', async () => {
	const myTool = tool({
		description: 'returns a number',
		inputSchema: z.object({ n: z.number().int() }),
		execute: ({ n }) => n,
	});

	let captured: any;
	const fakeFetch = async (_url: string | URL | Request, init?: RequestInit) => {
		captured = JSON.parse(init?.body as string);
		return sseResponse([
			JSON.stringify({ choices: [{ index: 0, delta: { content: 'k' } }] }),
			JSON.stringify({ choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] }),
			'[DONE]',
		]);
	};

	const adapter = openaiCompatible('test-model', {
		name: 'compat',
		baseUrl: 'https://compat.example/v1',
		apiKey: 'test',
		fetch: fakeFetch as typeof fetch,
	});

	for await (const _ of chat({
		adapter,
		messages: [user('hi')],
		tools: { my_tool: myTool },
	})) {
		// drain
	}

	const tools = captured?.tools as Array<{ function: { name: string; parameters: any } }>;
	assert.equal(tools.length, 1);
	assert.equal(tools[0]?.function.name, 'my_tool');
	const params = tools[0]?.function.parameters;
	assert.equal(params.type, 'object');
	assert.ok(params.properties.n);
});

test('openaiCompatible structuredOutput uses Chat Completions response_format', async () => {
	let capturedSchema: any;
	const fakeFetch = async (_url: string | URL | Request, init?: RequestInit) => {
		const body = JSON.parse(init?.body as string);
		if (body.response_format) {
			capturedSchema = body.response_format.json_schema.schema;
			return new Response(
				JSON.stringify({
					choices: [{ message: { content: JSON.stringify({ name: 'cat', alias: null }) } }],
				}),
				{ status: 200, headers: { 'content-type': 'application/json' } },
			);
		}
		return sseResponse([
			JSON.stringify({ choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] }),
			'[DONE]',
		]);
	};

	const adapter = openaiCompatible('test-model', {
		name: 'compat',
		baseUrl: 'https://compat.example/v1',
		apiKey: 'test',
		fetch: fakeFetch as typeof fetch,
	});

	const { object } = await generateObject({
		adapter,
		messages: [user('name something')],
		schema: z.object({ name: z.string(), alias: z.string().optional() }),
	});

	assert.equal(capturedSchema.additionalProperties, false);
	assert.deepEqual(capturedSchema.required, ['name', 'alias']);
	assert.deepEqual(capturedSchema.properties.alias.type, ['string', 'null']);
	assert.equal(object.name, 'cat');
	assert.equal(object.alias, undefined);
});
