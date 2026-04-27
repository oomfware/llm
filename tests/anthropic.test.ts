import assert from 'node:assert/strict';
import { test } from 'node:test';

import { z } from 'zod';

import { chat } from '../src/chat.ts';
import { system, user } from '../src/messages.ts';
import { anthropic } from '../src/providers/anthropic.ts';
import { tool } from '../src/tool.ts';

import { collect, sseResponse } from './_helpers.ts';

test('parses an anthropic text stream and emits finish with usage', async () => {
	const events = [
		JSON.stringify({
			type: 'message_start',
			index: 0,
			message: { usage: { input_tokens: 12, output_tokens: 0 } },
		}),
		JSON.stringify({
			type: 'content_block_start',
			index: 0,
			content_block: { type: 'text', text: '' },
		}),
		JSON.stringify({
			type: 'content_block_delta',
			index: 0,
			delta: { type: 'text_delta', text: 'hi ' },
		}),
		JSON.stringify({
			type: 'content_block_delta',
			index: 0,
			delta: { type: 'text_delta', text: 'there' },
		}),
		JSON.stringify({ type: 'content_block_stop', index: 0 }),
		JSON.stringify({
			type: 'message_delta',
			index: 0,
			delta: { stop_reason: 'end_turn' },
			usage: { output_tokens: 5 },
		}),
		JSON.stringify({ type: 'message_stop', index: 0 }),
	];

	const fakeFetch = async () => sseResponse(events);
	const adapter = anthropic('claude-sonnet-4-5', {
		apiKey: 'test',
		fetch: fakeFetch as typeof fetch,
	});

	const chunks = await collect(chat({ adapter, messages: [user('hi')] }));

	const text = chunks
		.filter((c) => c.type === 'text-delta')
		.map((c) => c.delta)
		.join('');
	assert.equal(text, 'hi there');

	const finish = chunks.at(-1);
	assert.equal(finish?.type, 'finish');
	if (finish?.type === 'finish') {
		assert.equal(finish.reason, 'stop');
		assert.equal(finish.usage?.inputTokens, 12);
		assert.equal(finish.usage?.outputTokens, 5);
	}
});

test('parses an anthropic tool_use stream and feeds tool result back', async () => {
	const getWeather = tool({
		inputSchema: z.object({ city: z.string() }),
		execute: ({ city }) => `sunny in ${city}`,
	});

	let callCount = 0;
	const fakeFetch = async (_url: string | URL | Request, init?: RequestInit) => {
		callCount++;
		if (callCount === 1) {
			return sseResponse([
				JSON.stringify({
					type: 'message_start',
					index: 0,
					message: { usage: { input_tokens: 1, output_tokens: 0 } },
				}),
				JSON.stringify({
					type: 'content_block_start',
					index: 0,
					content_block: { type: 'tool_use', id: 'toolu_1', name: 'get_weather' },
				}),
				JSON.stringify({
					type: 'content_block_delta',
					index: 0,
					delta: { type: 'input_json_delta', partial_json: '{"city":' },
				}),
				JSON.stringify({
					type: 'content_block_delta',
					index: 0,
					delta: { type: 'input_json_delta', partial_json: '"paris"}' },
				}),
				JSON.stringify({ type: 'content_block_stop', index: 0 }),
				JSON.stringify({
					type: 'message_delta',
					index: 0,
					delta: { stop_reason: 'tool_use' },
					usage: { output_tokens: 8 },
				}),
				JSON.stringify({ type: 'message_stop', index: 0 }),
			]);
		}

		const body = JSON.parse(init?.body as string) as {
			messages: {
				role: string;
				content: { type: string; tool_use_id?: string; content?: string }[];
			}[];
		};
		const last = body.messages.at(-1)!;
		assert.equal(last.role, 'user');
		const block = last.content.find((b) => b.type === 'tool_result');
		assert.ok(block, 'expected a tool_result block');
		assert.equal(block?.tool_use_id, 'toolu_1');
		assert.equal(block?.content, 'sunny in paris');

		return sseResponse([
			JSON.stringify({
				type: 'message_start',
				index: 0,
				message: { usage: { input_tokens: 5, output_tokens: 0 } },
			}),
			JSON.stringify({
				type: 'content_block_start',
				index: 0,
				content_block: { type: 'text', text: '' },
			}),
			JSON.stringify({
				type: 'content_block_delta',
				index: 0,
				delta: { type: 'text_delta', text: 'paris is sunny.' },
			}),
			JSON.stringify({ type: 'content_block_stop', index: 0 }),
			JSON.stringify({
				type: 'message_delta',
				index: 0,
				delta: { stop_reason: 'end_turn' },
				usage: { output_tokens: 3 },
			}),
			JSON.stringify({ type: 'message_stop', index: 0 }),
		]);
	};

	const adapter = anthropic('claude-sonnet-4-5', {
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
	if (toolStart?.type === 'tool-call-start') {
		assert.equal(toolStart.name, 'get_weather');
		assert.equal(toolStart.id, 'toolu_1');
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
	assert.equal(finalText, 'paris is sunny.');
});

for (const tc of [
	{
		name: 'captures thinking blocks with signatures and round-trips them on resume',
		firstTurnThinking: {
			content_block: { type: 'thinking', thinking: '', signature: '' },
			deltas: [
				{ type: 'thinking_delta', thinking: 'considering...' },
				{ type: 'signature_delta', signature: 'sig-123' },
			],
		},
		expectedReplayBlock: { type: 'thinking', field: 'signature', value: 'sig-123' },
	},
	{
		name: 'round-trips redacted_thinking blocks verbatim on resume',
		firstTurnThinking: {
			content_block: { type: 'redacted_thinking', data: 'opaque-blob' },
			deltas: [],
		},
		expectedReplayBlock: { type: 'redacted_thinking', field: 'data', value: 'opaque-blob' },
	},
]) {
	test(tc.name, async () => {
		let callCount = 0;
		const captures: any[] = [];
		const fakeFetch = async (_url: string | URL | Request, init?: RequestInit) => {
			callCount++;
			captures.push(JSON.parse(init?.body as string));
			if (callCount === 1) {
				return sseResponse([
					JSON.stringify({ type: 'message_start', index: 0 }),
					JSON.stringify({
						type: 'content_block_start',
						index: 0,
						content_block: tc.firstTurnThinking.content_block,
					}),
					...tc.firstTurnThinking.deltas.map((delta) =>
						JSON.stringify({ type: 'content_block_delta', index: 0, delta }),
					),
					JSON.stringify({ type: 'content_block_stop', index: 0 }),
					JSON.stringify({
						type: 'content_block_start',
						index: 1,
						content_block: { type: 'tool_use', id: 'tu_1', name: 'noop' },
					}),
					JSON.stringify({
						type: 'content_block_delta',
						index: 1,
						delta: { type: 'input_json_delta', partial_json: '{}' },
					}),
					JSON.stringify({ type: 'content_block_stop', index: 1 }),
					JSON.stringify({ type: 'message_delta', index: 0, delta: { stop_reason: 'tool_use' } }),
				]);
			}
			return sseResponse([
				JSON.stringify({ type: 'message_start', index: 0 }),
				JSON.stringify({ type: 'content_block_start', index: 0, content_block: { type: 'text' } }),
				JSON.stringify({
					type: 'content_block_delta',
					index: 0,
					delta: { type: 'text_delta', text: 'done' },
				}),
				JSON.stringify({ type: 'content_block_stop', index: 0 }),
				JSON.stringify({ type: 'message_delta', index: 0, delta: { stop_reason: 'end_turn' } }),
			]);
		};

		const noop = tool({ inputSchema: z.object({}), execute: () => 'k' });
		const adapter = anthropic('claude-sonnet-4-5', { apiKey: 'test', fetch: fakeFetch as typeof fetch });

		await collect(
			chat({
				adapter,
				messages: [user('think then act')],
				tools: { noop },
				providerOptions: { thinking: { type: 'enabled', budgetTokens: 4096 } },
			}),
		);

		assert.equal(callCount, 2);
		const second = captures[1];
		const assistantMsg = second.messages.find((m: any) => m.role === 'assistant');
		assert.ok(assistantMsg, 'expected an assistant message in the replayed conversation');
		const blocks = assistantMsg.content as Record<string, unknown>[];
		const block = blocks.find((b) => b.type === tc.expectedReplayBlock.type);
		assert.ok(block, `expected a ${tc.expectedReplayBlock.type} block on replay`);
		assert.equal(block?.[tc.expectedReplayBlock.field], tc.expectedReplayBlock.value);
	});
}

test('emits reasoning-delta for thinking content blocks', async () => {
	const events = [
		JSON.stringify({ type: 'message_start', index: 0 }),
		JSON.stringify({
			type: 'content_block_start',
			index: 0,
			content_block: { type: 'thinking' },
		}),
		JSON.stringify({
			type: 'content_block_delta',
			index: 0,
			delta: { type: 'thinking_delta', thinking: 'considering...' },
		}),
		JSON.stringify({ type: 'content_block_stop', index: 0 }),
		JSON.stringify({
			type: 'content_block_start',
			index: 1,
			content_block: { type: 'text' },
		}),
		JSON.stringify({
			type: 'content_block_delta',
			index: 1,
			delta: { type: 'text_delta', text: 'answer' },
		}),
		JSON.stringify({ type: 'content_block_stop', index: 1 }),
		JSON.stringify({ type: 'message_delta', index: 0, delta: { stop_reason: 'end_turn' } }),
	];

	const fakeFetch = async () => sseResponse(events);
	const adapter = anthropic('claude-sonnet-4-5', {
		apiKey: 'test',
		fetch: fakeFetch as typeof fetch,
	});

	const chunks = await collect(chat({ adapter, messages: [user('think')] }));

	const reasoning = chunks
		.filter((c) => c.type === 'reasoning-delta')
		.map((c) => c.delta)
		.join('');
	assert.equal(reasoning, 'considering...');

	const text = chunks
		.filter((c) => c.type === 'text-delta')
		.map((c) => c.delta)
		.join('');
	assert.equal(text, 'answer');
});

test('lifts system messages out of the message array', async () => {
	let captured: Record<string, unknown> | undefined;
	const fakeFetch = async (_url: string | URL | Request, init?: RequestInit) => {
		captured = JSON.parse(init?.body as string);
		return sseResponse([
			JSON.stringify({ type: 'message_start', index: 0 }),
			JSON.stringify({
				type: 'content_block_start',
				index: 0,
				content_block: { type: 'text' },
			}),
			JSON.stringify({
				type: 'content_block_delta',
				index: 0,
				delta: { type: 'text_delta', text: 'k' },
			}),
			JSON.stringify({ type: 'content_block_stop', index: 0 }),
			JSON.stringify({ type: 'message_delta', index: 0, delta: { stop_reason: 'end_turn' } }),
		]);
	};

	const adapter = anthropic('claude-haiku-4-5', {
		apiKey: 'test',
		fetch: fakeFetch as typeof fetch,
	});

	for await (const _ of chat({
		adapter,
		messages: [system('you are terse.'), user('hi')],
	})) {
		// drain
	}

	const systemBlocks = captured?.system as { type: string; text: string }[];
	assert.ok(Array.isArray(systemBlocks));
	assert.equal(systemBlocks[0]?.text, 'you are terse.');
	const messages = captured?.messages as Array<{ role: string }>;
	assert.equal(messages.length, 1);
	assert.equal(messages[0]?.role, 'user');
});

test('captures cache_creation/read tokens from anthropic usage', async () => {
	const events = [
		JSON.stringify({
			type: 'message_start',
			index: 0,
			message: {
				usage: {
					input_tokens: 12,
					output_tokens: 0,
					cache_creation_input_tokens: 100,
					cache_read_input_tokens: 50,
				},
			},
		}),
		JSON.stringify({
			type: 'content_block_start',
			index: 0,
			content_block: { type: 'text' },
		}),
		JSON.stringify({
			type: 'content_block_delta',
			index: 0,
			delta: { type: 'text_delta', text: 'hi' },
		}),
		JSON.stringify({ type: 'content_block_stop', index: 0 }),
		JSON.stringify({
			type: 'message_delta',
			index: 0,
			delta: { stop_reason: 'end_turn' },
			usage: { output_tokens: 5 },
		}),
	];

	const fakeFetch = async () => sseResponse(events);
	const adapter = anthropic('claude-sonnet-4-5', {
		apiKey: 'test',
		fetch: fakeFetch as typeof fetch,
	});

	const chunks = await collect(chat({ adapter, messages: [user('hi')] }));

	const finish = chunks.at(-1);
	if (finish?.type === 'finish') {
		assert.equal(finish.usage?.cacheCreationInputTokens, 100);
		assert.equal(finish.usage?.cacheReadInputTokens, 50);
	}
});

test('cacheControl on a part stamps cache_control on the wire block', async () => {
	let captured: any;
	const fakeFetch = async (_url: string | URL | Request, init?: RequestInit) => {
		captured = JSON.parse(init?.body as string);
		return sseResponse([
			JSON.stringify({ type: 'message_start', index: 0 }),
			JSON.stringify({
				type: 'content_block_start',
				index: 0,
				content_block: { type: 'text' },
			}),
			JSON.stringify({
				type: 'content_block_delta',
				index: 0,
				delta: { type: 'text_delta', text: 'k' },
			}),
			JSON.stringify({ type: 'content_block_stop', index: 0 }),
			JSON.stringify({ type: 'message_delta', index: 0, delta: { stop_reason: 'end_turn' } }),
		]);
	};

	const adapter = anthropic('claude-sonnet-4-5', {
		apiKey: 'test',
		fetch: fakeFetch as typeof fetch,
	});

	for await (const _ of chat({
		adapter,
		messages: [
			{
				role: 'user',
				content: [
					{
						type: 'text',
						text: 'cache me',
						providerMetadata: { anthropic: { cacheControl: { type: 'ephemeral', ttl: '1h' } } },
					},
				],
			},
		],
	})) {
		// drain
	}

	const block = captured?.messages?.[0]?.content?.[0];
	assert.equal(block?.type, 'text');
	assert.deepEqual(block?.cache_control, { type: 'ephemeral', ttl: '1h' });
});

test('trims trailing whitespace on the last text of a prefilled assistant turn', async () => {
	let captured: any;
	const fakeFetch = async (_url: string | URL | Request, init?: RequestInit) => {
		captured = JSON.parse(init?.body as string);
		return sseResponse([
			JSON.stringify({ type: 'message_start', index: 0 }),
			JSON.stringify({
				type: 'content_block_start',
				index: 0,
				content_block: { type: 'text' },
			}),
			JSON.stringify({
				type: 'content_block_delta',
				index: 0,
				delta: { type: 'text_delta', text: 'continued' },
			}),
			JSON.stringify({ type: 'content_block_stop', index: 0 }),
			JSON.stringify({ type: 'message_delta', index: 0, delta: { stop_reason: 'end_turn' } }),
		]);
	};

	const adapter = anthropic('claude-sonnet-4-5', {
		apiKey: 'test',
		fetch: fakeFetch as typeof fetch,
	});

	for await (const _ of chat({
		adapter,
		messages: [
			user('start a sentence:'),
			{
				role: 'assistant',
				content: [{ type: 'text', text: 'the quick brown fox  \n\t' }],
			},
		],
	})) {
		// drain
	}

	const blocks = captured?.messages?.[1]?.content as { type: string; text: string }[];
	assert.equal(blocks[0]?.text, 'the quick brown fox');
});

test('does not trim trailing whitespace when the assistant turn is not the last message', async () => {
	let captured: any;
	const fakeFetch = async (_url: string | URL | Request, init?: RequestInit) => {
		captured = JSON.parse(init?.body as string);
		return sseResponse([
			JSON.stringify({ type: 'message_start', index: 0 }),
			JSON.stringify({ type: 'content_block_start', index: 0, content_block: { type: 'text' } }),
			JSON.stringify({
				type: 'content_block_delta',
				index: 0,
				delta: { type: 'text_delta', text: 'ok' },
			}),
			JSON.stringify({ type: 'content_block_stop', index: 0 }),
			JSON.stringify({ type: 'message_delta', index: 0, delta: { stop_reason: 'end_turn' } }),
		]);
	};

	const adapter = anthropic('claude-sonnet-4-5', { apiKey: 'test', fetch: fakeFetch as typeof fetch });

	for await (const _ of chat({
		adapter,
		messages: [
			user('a'),
			{ role: 'assistant', content: [{ type: 'text', text: 'middle reply  ' }] },
			user('b'),
		],
	})) {
		// drain
	}

	const blocks = captured?.messages?.[1]?.content as { type: string; text: string }[];
	// trailing whitespace preserved because this assistant turn isn't the last message
	assert.equal(blocks[0]?.text, 'middle reply  ');
});

test('drops reasoning parts that lack an anthropic signature on resume', async () => {
	let captured: any;
	const fakeFetch = async (_url: string | URL | Request, init?: RequestInit) => {
		captured = JSON.parse(init?.body as string);
		return sseResponse([
			JSON.stringify({ type: 'message_start', index: 0 }),
			JSON.stringify({ type: 'content_block_start', index: 0, content_block: { type: 'text' } }),
			JSON.stringify({
				type: 'content_block_delta',
				index: 0,
				delta: { type: 'text_delta', text: 'ok' },
			}),
			JSON.stringify({ type: 'content_block_stop', index: 0 }),
			JSON.stringify({ type: 'message_delta', index: 0, delta: { stop_reason: 'end_turn' } }),
		]);
	};

	const adapter = anthropic('claude-sonnet-4-5', { apiKey: 'test', fetch: fakeFetch as typeof fetch });

	for await (const _ of chat({
		adapter,
		messages: [
			user('hi'),
			{
				role: 'assistant',
				content: [
					// no signature — should be dropped on the wire
					{ type: 'reasoning', text: 'thinking' },
					{ type: 'text', text: 'and the answer' },
				],
			},
			user('next'),
		],
	})) {
		// drain
	}

	const assistantBlocks = captured?.messages?.[1]?.content as { type: string }[];
	assert.equal(
		assistantBlocks.find((b) => b.type === 'thinking'),
		undefined,
	);
	assert.ok(assistantBlocks.find((b) => b.type === 'text'));
});

test('round-trips ToolResultPart.isError as is_error: true on the wire', async () => {
	let captured: any;
	const fakeFetch = async (_url: string | URL | Request, init?: RequestInit) => {
		captured = JSON.parse(init?.body as string);
		return sseResponse([
			JSON.stringify({ type: 'message_start', index: 0 }),
			JSON.stringify({ type: 'content_block_start', index: 0, content_block: { type: 'text' } }),
			JSON.stringify({
				type: 'content_block_delta',
				index: 0,
				delta: { type: 'text_delta', text: 'noted' },
			}),
			JSON.stringify({ type: 'content_block_stop', index: 0 }),
			JSON.stringify({ type: 'message_delta', index: 0, delta: { stop_reason: 'end_turn' } }),
		]);
	};

	const adapter = anthropic('claude-sonnet-4-5', { apiKey: 'test', fetch: fakeFetch as typeof fetch });

	for await (const _ of chat({
		adapter,
		messages: [
			user('hi'),
			{
				role: 'assistant',
				content: [{ type: 'tool-call', id: 'tu_1', name: 'noop', arguments: '{}' }],
			},
			{
				role: 'tool',
				content: [{ type: 'tool-result', toolCallId: 'tu_1', output: 'boom', isError: true }],
			},
		],
	})) {
		// drain
	}

	// the tool result is sent as a user message containing tool_result blocks
	const toolUserMsg = captured?.messages?.find(
		(m: any) => m.role === 'user' && Array.isArray(m.content) && m.content[0]?.type === 'tool_result',
	);
	assert.ok(toolUserMsg);
	assert.equal(toolUserMsg.content[0].is_error, true);
});
