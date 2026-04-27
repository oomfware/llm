import assert from 'node:assert/strict';
import { test } from 'node:test';

import { z } from 'zod';

import { chat } from '../src/chat.ts';
import { user } from '../src/messages.ts';
import { openai } from '../src/providers/openai.ts';
import { tool } from '../src/tool.ts';

import { collect, sseResponse } from './_helpers.ts';

test('parses an OpenAI text stream and normalizes finish reason', async () => {
	const events = [
		JSON.stringify({
			type: 'response.output_item.added',
			item: { type: 'message', id: 'msg_1', role: 'assistant', content: [] },
		}),
		JSON.stringify({ type: 'response.output_text.delta', item_id: 'msg_1', delta: 'hello ' }),
		JSON.stringify({ type: 'response.output_text.delta', item_id: 'msg_1', delta: 'world' }),
		JSON.stringify({
			type: 'response.output_item.done',
			item: { type: 'message', id: 'msg_1', role: 'assistant', status: 'completed', content: [] },
		}),
		JSON.stringify({
			type: 'response.completed',
			response: {
				status: 'completed',
				incomplete_details: null,
				usage: { input_tokens: 10, output_tokens: 2 },
			},
		}),
		'[DONE]',
	];

	const fakeFetch = async () => sseResponse(events);

	const adapter = openai('gpt-4o', { apiKey: 'test', fetch: fakeFetch as typeof fetch });

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

test('parses an OpenAI tool-call stream across deltas', async () => {
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
					type: 'response.output_item.added',
					item: {
						type: 'function_call',
						id: 'fc_1',
						call_id: 'call_1',
						name: 'get_weather',
						arguments: '',
					},
				}),
				JSON.stringify({
					type: 'response.function_call_arguments.delta',
					item_id: 'fc_1',
					delta: '{"city":',
				}),
				JSON.stringify({
					type: 'response.function_call_arguments.delta',
					item_id: 'fc_1',
					delta: '"paris"}',
				}),
				JSON.stringify({
					type: 'response.function_call_arguments.done',
					item_id: 'fc_1',
					name: 'get_weather',
					arguments: '{"city":"paris"}',
				}),
				JSON.stringify({
					type: 'response.completed',
					response: {
						status: 'completed',
						incomplete_details: null,
						usage: { input_tokens: 1, output_tokens: 8 },
					},
				}),
				'[DONE]',
			]);
		}

		// second call - verify the tool result was fed back as a function-call output item
		const body = JSON.parse(init?.body as string) as {
			input: { type: string; call_id?: string; output?: string }[];
		};
		const lastMessage = body.input.at(-1)!;
		assert.equal(lastMessage.type, 'function_call_output');
		assert.equal(lastMessage.call_id, 'call_1');
		assert.equal(lastMessage.output, 'sunny in paris');

		return sseResponse([
			JSON.stringify({
				type: 'response.output_item.added',
				item: { type: 'message', id: 'msg_2', role: 'assistant', content: [] },
			}),
			JSON.stringify({ type: 'response.output_text.delta', item_id: 'msg_2', delta: 'reported.' }),
			JSON.stringify({
				type: 'response.output_item.done',
				item: { type: 'message', id: 'msg_2', role: 'assistant', status: 'completed', content: [] },
			}),
			JSON.stringify({
				type: 'response.completed',
				response: { status: 'completed', incomplete_details: null },
			}),
			'[DONE]',
		]);
	};

	const adapter = openai('gpt-4o', { apiKey: 'test', fetch: fakeFetch as typeof fetch });

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

test('forwards reasoningEffort for reasoning models and includes encrypted_content', async () => {
	let captured: Record<string, unknown> | undefined;
	const fakeFetch = async (_url: string | URL | Request, init?: RequestInit) => {
		captured = JSON.parse(init?.body as string);
		return sseResponse([
			JSON.stringify({
				type: 'response.output_item.added',
				item: { type: 'message', id: 'msg_1', role: 'assistant', content: [] },
			}),
			JSON.stringify({ type: 'response.output_text.delta', item_id: 'msg_1', delta: 'ok' }),
			JSON.stringify({
				type: 'response.output_item.done',
				item: { type: 'message', id: 'msg_1', role: 'assistant', status: 'completed', content: [] },
			}),
			JSON.stringify({
				type: 'response.completed',
				response: { status: 'completed', incomplete_details: null },
			}),
			'[DONE]',
		]);
	};

	const adapter = openai('gpt-5', { apiKey: 'test', fetch: fakeFetch as typeof fetch });

	const stream = chat({
		adapter,
		messages: [user('hi')],
		providerOptions: { reasoningEffort: 'high' },
	});
	for await (const _ of stream) {
		// drain
	}

	assert.deepEqual(captured?.reasoning, { effort: 'high' });
	assert.equal(captured?.model, 'gpt-5');
	assert.deepEqual(captured?.include, ['reasoning.encrypted_content']);
});

test('round-trips reasoning items with id + encryptedContent on resume', async () => {
	let callCount = 0;
	const captures: any[] = [];
	const fakeFetch = async (_url: string | URL | Request, init?: RequestInit) => {
		callCount++;
		const body = JSON.parse(init?.body as string);
		captures.push(body);
		if (callCount === 1) {
			return sseResponse([
				JSON.stringify({
					type: 'response.output_item.added',
					item: { type: 'reasoning', id: 'rs_1' },
				}),
				JSON.stringify({
					type: 'response.reasoning_text.delta',
					item_id: 'rs_1',
					delta: 'thinking',
				}),
				JSON.stringify({
					type: 'response.output_item.done',
					item: {
						type: 'reasoning',
						id: 'rs_1',
						summary: [],
						content: [],
						encrypted_content: 'ENC123',
						status: 'completed',
					},
				}),
				JSON.stringify({
					type: 'response.output_item.added',
					item: { type: 'function_call', id: 'fc_1', call_id: 'call_1', name: 'noop', arguments: '' },
				}),
				JSON.stringify({
					type: 'response.function_call_arguments.delta',
					item_id: 'fc_1',
					delta: '{}',
				}),
				JSON.stringify({
					type: 'response.function_call_arguments.done',
					item_id: 'fc_1',
					name: 'noop',
					arguments: '{}',
				}),
				JSON.stringify({
					type: 'response.completed',
					response: { status: 'completed', incomplete_details: null },
				}),
				'[DONE]',
			]);
		}
		return sseResponse([
			JSON.stringify({
				type: 'response.output_item.added',
				item: { type: 'message', id: 'msg_2', role: 'assistant', content: [] },
			}),
			JSON.stringify({ type: 'response.output_text.delta', item_id: 'msg_2', delta: 'done' }),
			JSON.stringify({
				type: 'response.output_item.done',
				item: { type: 'message', id: 'msg_2', role: 'assistant', status: 'completed', content: [] },
			}),
			JSON.stringify({
				type: 'response.completed',
				response: { status: 'completed', incomplete_details: null },
			}),
			'[DONE]',
		]);
	};

	const noop = tool({
		inputSchema: z.object({}),
		execute: () => 'ok',
	});

	const adapter = openai('gpt-5', { apiKey: 'test', fetch: fakeFetch as typeof fetch });

	await collect(
		chat({
			adapter,
			messages: [user('think')],
			tools: { noop },
			providerOptions: { reasoningEffort: 'medium' },
		}),
	);

	assert.equal(callCount, 2);
	const second = captures[1];
	const items = second.input as { type: string; id?: string; encrypted_content?: string }[];
	const reasoningItem = items.find((i) => i.type === 'reasoning');
	assert.ok(reasoningItem, 'expected the reasoning item to be replayed in input');
	assert.equal(reasoningItem?.id, 'rs_1');
	assert.equal(reasoningItem?.encrypted_content, 'ENC123');
});

for (const itemType of ['image_generation_call', 'mcp_call', 'file_search_call']) {
	test(`fails fast on unsupported output item type: ${itemType}`, async () => {
		const events = [
			JSON.stringify({
				type: 'response.output_item.added',
				item: { type: itemType, id: 'x_1' },
			}),
			'[DONE]',
		];
		const fakeFetch = async () => sseResponse(events);
		const adapter = openai('gpt-4o', { apiKey: 'test', fetch: fakeFetch as typeof fetch });

		const chunks = await collect(chat({ adapter, messages: [user('go')] }));
		const error = chunks.find((c) => c.type === 'error');
		assert.ok(error, 'expected an error chunk for the unsupported item');
		if (error?.type === 'error') {
			assert.match((error.error as Error).message, /unsupported output item type/);
		}
	});
}

test('round-trips assistant text messages with id + content on resume', async () => {
	let callCount = 0;
	const captures: any[] = [];
	const completedTextEvents = [
		JSON.stringify({
			type: 'response.output_item.added',
			item: { type: 'message', id: 'msg_1', role: 'assistant', content: [] },
		}),
		JSON.stringify({ type: 'response.output_text.delta', item_id: 'msg_1', delta: 'hello' }),
		JSON.stringify({
			type: 'response.output_item.done',
			item: {
				type: 'message',
				id: 'msg_1',
				role: 'assistant',
				status: 'completed',
				content: [{ type: 'output_text', text: 'hello', annotations: [] }],
			},
		}),
		JSON.stringify({
			type: 'response.completed',
			response: { status: 'completed', incomplete_details: null },
		}),
		'[DONE]',
	];
	const fakeFetch = async (_url: string | URL | Request, init?: RequestInit) => {
		callCount++;
		captures.push(JSON.parse(init?.body as string));
		return sseResponse(completedTextEvents);
	};

	const adapter = openai('gpt-4o', { apiKey: 'test', fetch: fakeFetch as typeof fetch });

	// resumed conversation: assistant turn carries an openai id from a prior call
	for await (const _ of chat({
		adapter,
		messages: [
			user('hi'),
			{
				role: 'assistant',
				content: [
					{
						type: 'text',
						text: 'hello',
						providerMetadata: { openai: { id: 'msg_prior', status: 'completed' } },
					},
				],
			},
			user('continue'),
		],
	})) {
		// drain
	}

	assert.equal(callCount, 1);
	const replayedAssistant = captures[0]?.input?.find(
		(item: any) => item.type === 'message' && item.role === 'assistant',
	);
	assert.ok(replayedAssistant);
	assert.equal(replayedAssistant.id, 'msg_prior');
	assert.equal(replayedAssistant.status, 'completed');
});

test('honors includeEncryptedReasoning: false even when reasoningEffort is set', async () => {
	let captured: Record<string, unknown> | undefined;
	const fakeFetch = async (_url: string | URL | Request, init?: RequestInit) => {
		captured = JSON.parse(init?.body as string);
		return sseResponse([
			JSON.stringify({
				type: 'response.output_item.added',
				item: { type: 'message', id: 'msg_1', role: 'assistant', content: [] },
			}),
			JSON.stringify({ type: 'response.output_text.delta', item_id: 'msg_1', delta: 'ok' }),
			JSON.stringify({
				type: 'response.output_item.done',
				item: { type: 'message', id: 'msg_1', role: 'assistant', status: 'completed', content: [] },
			}),
			JSON.stringify({ type: 'response.completed', response: { status: 'completed' } }),
			'[DONE]',
		]);
	};

	const adapter = openai('gpt-5', { apiKey: 'test', fetch: fakeFetch as typeof fetch });

	for await (const _ of chat({
		adapter,
		messages: [user('hi')],
		providerOptions: { reasoningEffort: 'high', includeEncryptedReasoning: false },
	})) {
		// drain
	}

	assert.deepEqual(captured?.reasoning, { effort: 'high' });
	assert.equal(captured?.include, undefined);
});

test('captures cached_tokens and reasoning_tokens in usage', async () => {
	const events = [
		JSON.stringify({
			type: 'response.output_item.added',
			item: { type: 'message', id: 'msg_1', role: 'assistant', content: [] },
		}),
		JSON.stringify({ type: 'response.output_text.delta', item_id: 'msg_1', delta: 'ok' }),
		JSON.stringify({
			type: 'response.output_item.done',
			item: { type: 'message', id: 'msg_1', role: 'assistant', status: 'completed', content: [] },
		}),
		JSON.stringify({
			type: 'response.completed',
			response: {
				status: 'completed',
				incomplete_details: null,
				usage: {
					input_tokens: 100,
					output_tokens: 50,
					input_tokens_details: { cached_tokens: 60 },
					output_tokens_details: { reasoning_tokens: 30 },
				},
			},
		}),
		'[DONE]',
	];
	const fakeFetch = async () => sseResponse(events);
	const adapter = openai('gpt-4o', { apiKey: 'test', fetch: fakeFetch as typeof fetch });

	const chunks = await collect(chat({ adapter, messages: [user('hi')] }));
	const finish = chunks.at(-1);
	if (finish?.type === 'finish') {
		assert.equal(finish.usage?.cacheReadInputTokens, 60);
		assert.equal(finish.usage?.reasoningTokens, 30);
	}
});

test('surfaces a non-2xx response as an error chunk', async () => {
	const fakeFetch = async () => new Response('bad request', { status: 400, statusText: 'Bad Request' });

	const adapter = openai('gpt-4o', { apiKey: 'test', fetch: fakeFetch as typeof fetch });

	const chunks = await collect(chat({ adapter, messages: [user('hi')] }));

	const error = chunks.find((c) => c.type === 'error');
	assert.ok(error, 'expected an error chunk');
	if (error?.type === 'error') {
		assert.match((error.error as Error).message, /openai: 400/);
	}
});

test('sends the tool jsonSchema (not the standard schema object) on the wire', async () => {
	const myTool = tool({
		description: 'returns a number',
		inputSchema: z.object({ n: z.number().int() }),
		execute: ({ n }) => n,
	});

	let captured: any;
	const fakeFetch = async (_url: string | URL | Request, init?: RequestInit) => {
		captured = JSON.parse(init?.body as string);
		return sseResponse([
			JSON.stringify({
				type: 'response.output_item.added',
				item: { type: 'message', id: 'msg_1', role: 'assistant', content: [] },
			}),
			JSON.stringify({ type: 'response.output_text.delta', item_id: 'msg_1', delta: 'k' }),
			JSON.stringify({
				type: 'response.output_item.done',
				item: { type: 'message', id: 'msg_1', role: 'assistant', status: 'completed', content: [] },
			}),
			JSON.stringify({
				type: 'response.completed',
				response: { status: 'completed', incomplete_details: null },
			}),
			'[DONE]',
		]);
	};

	const adapter = openai('gpt-4o', { apiKey: 'test', fetch: fakeFetch as typeof fetch });

	for await (const _ of chat({
		adapter,
		messages: [user('hi')],
		tools: { my_tool: myTool },
	})) {
		// drain
	}

	const tools = captured?.tools as Array<{ name: string; parameters: any }>;
	assert.equal(tools.length, 1);
	assert.equal(tools[0]?.name, 'my_tool');
	const params = tools[0]?.parameters;
	assert.equal(params.type, 'object');
	assert.ok(params.properties.n);
});
