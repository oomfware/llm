import assert from 'node:assert/strict';
import { test } from 'node:test';

import { z } from 'zod';

import { combineStrategies, maxIterations, untilFinishReason } from '../src/agent-loop-strategies.ts';
import { chat } from '../src/chat.ts';
import { ToolInputValidationError, ToolOutputValidationError, UnknownToolError } from '../src/errors.ts';
import { user } from '../src/messages.ts';
import { dummy } from '../src/providers/dummy.ts';
import { tool } from '../src/tool.ts';
import type { StreamChunk, ToolCallPart, ToolResultPart } from '../src/types.ts';

const collect = async <T extends StreamChunk<any>>(stream: AsyncIterable<T>): Promise<T[]> => {
	const out: T[] = [];
	for await (const chunk of stream) {
		out.push(chunk);
	}
	return out;
};

test('streams text and emits a stop finish', async () => {
	const adapter = dummy({ responses: [{ text: 'hello world' }] });

	const chunks = await collect(
		chat({
			adapter,
			messages: [user('hi')],
		}),
	);

	const text = chunks
		.filter((c) => c.type === 'text-delta')
		.map((c) => c.delta)
		.join('');
	assert.equal(text, 'hello world');

	const finish = chunks.at(-1);
	assert.equal(finish?.type, 'finish');
	if (finish?.type === 'finish') {
		assert.equal(finish.reason, 'stop');
	}

	assert.equal(adapter.calls.length, 1);
});

test('runs the tool loop: model -> tool -> model -> stop', async () => {
	const captured: { city: string }[] = [];
	const getWeather = tool({
		description: 'get weather for a city',
		inputSchema: z.object({ city: z.string() }),
		execute: ({ city }) => {
			captured.push({ city });
			return { city, conditions: 'sunny' };
		},
	});

	const adapter = dummy({
		responses: [
			{ toolCalls: [{ id: 'call_1', name: 'getWeather', arguments: { city: 'paris' } }] },
			{ text: 'it is sunny in paris.' },
		],
	});

	const chunks = await collect(
		chat({
			adapter,
			messages: [user('weather in paris?')],
			tools: { getWeather },
		}),
	);

	assert.deepEqual(captured, [{ city: 'paris' }]);
	assert.equal(adapter.calls.length, 2);

	const secondTurn = adapter.calls[1];
	const lastMessage = secondTurn?.messages.at(-1);
	assert.equal(lastMessage?.role, 'tool');
	if (lastMessage?.role === 'tool') {
		assert.equal(lastMessage.content[0]?.toolCallId, 'call_1');
	}

	const types = new Set(chunks.map((c) => c.type));
	assert.ok(types.has('tool-call-start'));
	assert.ok(types.has('tool-call-end'));
	assert.ok(types.has('tool-result'));

	// tool-call-end carries validated input
	const callEnd = chunks.find((c) => c.type === 'tool-call-end');
	assert.equal(callEnd?.type, 'tool-call-end');
	if (callEnd?.type === 'tool-call-end' && callEnd.name === 'getWeather') {
		assert.deepEqual(callEnd.input, { city: 'paris' });
	}

	// tool-result carries execute's return
	const result = chunks.find((c) => c.type === 'tool-result');
	if (result?.type === 'tool-result' && result.name === 'getWeather') {
		assert.deepEqual(result.result, { city: 'paris', conditions: 'sunny' });
	}

	const finish = chunks.at(-1);
	if (finish?.type === 'finish') {
		assert.equal(finish.reason, 'stop');
	}
});

for (const tc of [
	{ name: 'default strategy caps the loop at 5 iterations', strategy: undefined, expected: 5 },
	{ name: 'maxIterations(n) caps the loop at n iterations', strategy: maxIterations(3), expected: 3 },
	{
		name: 'combineStrategies stops when any strategy says so',
		strategy: combineStrategies(maxIterations(2), maxIterations(10)),
		expected: 2,
	},
]) {
	test(tc.name, async () => {
		const echo = tool({
			inputSchema: z.object({}),
			execute: () => 'ok',
		});

		const adapter = dummy({
			responses: Array.from({ length: 20 }, (_, i) => ({
				toolCalls: [{ id: `c_${i}`, name: 'echo', arguments: {} }],
			})),
		});

		const chunks = await collect(
			chat({
				adapter,
				messages: [user('loop')],
				tools: { echo },
				agentLoopStrategy: tc.strategy,
			}),
		);

		assert.equal(adapter.calls.length, tc.expected);

		const finish = chunks.at(-1);
		assert.equal(finish?.type, 'finish');
		if (finish?.type === 'finish') {
			assert.equal(finish.reason, 'length');
		}
	});
}

test('untilFinishReason can stop the loop after a tools turn', async () => {
	const echo = tool({
		inputSchema: z.object({}),
		execute: () => 'ok',
	});

	// every response would normally continue (toolCalls + reason 'tool-calls'),
	// but the strategy says "stop once you've seen 'tool-calls' once".
	const adapter = dummy({
		responses: Array.from({ length: 5 }, (_, i) => ({
			toolCalls: [{ id: `c_${i}`, name: 'echo', arguments: {} }],
		})),
	});

	const chunks = await collect(
		chat({
			adapter,
			messages: [user('go')],
			tools: { echo },
			agentLoopStrategy: untilFinishReason('tool-calls'),
		}),
	);

	// one model call, tools executed, strategy stops us before the next call
	assert.equal(adapter.calls.length, 1);

	const finish = chunks.at(-1);
	if (finish?.type === 'finish') {
		assert.equal(finish.reason, 'length');
	}
});

test('input schema validates and transforms arguments before execute', async () => {
	let received: unknown;
	const upper = tool({
		// schema's output applies a transform: lowercase -> uppercase
		inputSchema: z.object({
			s: z.string().transform((v) => v.toUpperCase()),
		}),
		execute: (input) => {
			received = input;
			return input.s;
		},
	});

	const adapter = dummy({
		responses: [{ toolCalls: [{ id: 'c', name: 'upper', arguments: { s: 'hi' } }] }, { text: 'done' }],
	});

	await collect(
		chat({
			adapter,
			messages: [user('go')],
			tools: { upper },
		}),
	);

	assert.deepEqual(received, { s: 'HI' });
});

test('input validation failure yields an error chunk', async () => {
	const strict = tool({
		inputSchema: z.object({ count: z.number().int().positive() }),
		execute: ({ count }) => count,
	});

	const adapter = dummy({
		// model returns an invalid value
		responses: [{ toolCalls: [{ id: 'c', name: 'strict', arguments: { count: -1 } }] }],
	});

	const chunks = await collect(
		chat({
			adapter,
			messages: [user('go')],
			tools: { strict },
		}),
	);

	const error = chunks.find((c) => c.type === 'error');
	assert.ok(error, 'expected an error chunk');
	if (error?.type === 'error') {
		assert.ok(error.error instanceof ToolInputValidationError);
	}
});

test('output validation failure yields an error chunk', async () => {
	const misbehaving = tool({
		inputSchema: z.object({}),
		outputSchema: z.object({ value: z.number() }),
		// returns the wrong shape on purpose
		execute: () => ({ value: 'not a number' }) as unknown as { value: number },
	});

	const adapter = dummy({
		responses: [{ toolCalls: [{ id: 'c', name: 'misbehaving', arguments: {} }] }],
	});

	const chunks = await collect(
		chat({
			adapter,
			messages: [user('go')],
			tools: { misbehaving },
		}),
	);

	const error = chunks.find((c) => c.type === 'error');
	assert.ok(error, 'expected an error chunk');
	if (error?.type === 'error') {
		assert.ok(error.error instanceof ToolOutputValidationError);
	}
	// no tool-result chunk should be emitted on validation failure
	assert.equal(
		chunks.find((c) => c.type === 'tool-result'),
		undefined,
	);
});

test('outputSchema transform applies before tool-result is yielded', async () => {
	const upper = tool({
		inputSchema: z.object({}),
		// transform on output: lowercase -> uppercase
		outputSchema: z.object({ s: z.string().transform((v) => v.toUpperCase()) }),
		execute: () => ({ s: 'hi' }),
	});

	const adapter = dummy({
		responses: [{ toolCalls: [{ id: 'c', name: 'upper', arguments: {} }] }, { text: 'done' }],
	});

	const chunks = await collect(
		chat({
			adapter,
			messages: [user('go')],
			tools: { upper },
		}),
	);

	const result = chunks.find((c) => c.type === 'tool-result');
	assert.ok(result);
	if (result?.type === 'tool-result' && result.name === 'upper') {
		assert.deepEqual(result.result, { s: 'HI' });
	}
});

test('unknown tool yields an error chunk and stops', async () => {
	const adapter = dummy({
		responses: [{ toolCalls: [{ id: 'c', name: 'nope', arguments: {} }] }],
	});

	const chunks = await collect(
		chat({
			adapter,
			messages: [user('go')],
		}),
	);

	const error = chunks.find((c) => c.type === 'error');
	assert.ok(error, 'expected an error chunk');
	if (error?.type === 'error') {
		assert.ok(error.error instanceof UnknownToolError);
		if (error.error instanceof UnknownToolError) {
			assert.equal(error.error.toolName, 'nope');
		}
	}
});

test('passes through error chunks from the adapter', async () => {
	const adapter = dummy({
		responses: [{ error: new Error('boom') }],
	});

	const chunks = await collect(
		chat({
			adapter,
			messages: [user('go')],
		}),
	);

	const error = chunks.find((c) => c.type === 'error');
	assert.ok(error);
	if (error?.type === 'error') {
		assert.equal((error.error as Error).message, 'boom');
	}
});

test('gated tool suspends with awaiting-approval; nothing executes', async () => {
	let executed = false;
	const sendEmail = tool({
		needsApproval: true,
		inputSchema: z.object({ to: z.string() }),
		execute: ({ to }) => {
			executed = true;
			return `sent to ${to}`;
		},
	});

	const adapter = dummy({
		responses: [{ toolCalls: [{ id: 'c1', name: 'sendEmail', arguments: { to: 'a@b.c' } }] }],
	});

	const chunks = await collect(
		chat({
			adapter,
			messages: [user('email a@b.c')],
			tools: { sendEmail },
		}),
	);

	assert.equal(executed, false);
	assert.equal(adapter.calls.length, 1);

	const approvalReq = chunks.find((c) => c.type === 'tool-approval-requested');
	assert.ok(approvalReq, 'expected a tool-approval-requested chunk');
	if (approvalReq?.type === 'tool-approval-requested' && approvalReq.name === 'sendEmail') {
		assert.equal(approvalReq.id, 'c1');
		assert.deepEqual(approvalReq.input, { to: 'a@b.c' });
	}

	const finish = chunks.at(-1);
	assert.equal(finish?.type, 'finish');
	if (finish?.type === 'finish') {
		assert.equal(finish.reason, 'awaiting-approval');
	}

	// no tool-result emitted while suspended
	assert.equal(
		chunks.find((c) => c.type === 'tool-result'),
		undefined,
	);
});

test('resume with approved=true executes without re-calling the model', async () => {
	let executed = false;
	const sendEmail = tool({
		needsApproval: true,
		inputSchema: z.object({ to: z.string() }),
		execute: ({ to }) => {
			executed = true;
			return `sent to ${to}`;
		},
	});

	// adapter only scripts the *follow-up* turn — the resumed call must not
	// hit the adapter for iteration 1.
	const adapter = dummy({
		responses: [{ text: 'done' }],
	});

	const chunks = await collect(
		chat({
			adapter,
			messages: [
				user('email a@b.c'),
				{
					role: 'assistant',
					content: [
						{
							type: 'tool-call',
							id: 'c1',
							name: 'sendEmail',
							arguments: JSON.stringify({ to: 'a@b.c' }),
							approval: { approved: true },
						},
					],
				},
			],
			tools: { sendEmail },
		}),
	);

	assert.equal(executed, true);
	// only one adapter call — for the post-execution turn, not for the resumed batch
	assert.equal(adapter.calls.length, 1);

	const result = chunks.find((c) => c.type === 'tool-result');
	assert.ok(result);
	if (result?.type === 'tool-result' && result.name === 'sendEmail') {
		assert.equal(result.result, 'sent to a@b.c');
	}

	const finish = chunks.at(-1);
	if (finish?.type === 'finish') {
		assert.equal(finish.reason, 'stop');
	}

	// the adapter saw the synthetic tool message before the follow-up turn
	const followUpMessages = adapter.calls[0]?.messages ?? [];
	const lastMsg = followUpMessages.at(-1);
	assert.equal(lastMsg?.role, 'tool');
	if (lastMsg?.role === 'tool') {
		const part = lastMsg.content[0];
		assert.equal(part?.toolCallId, 'c1');
		assert.match(part?.output ?? '', /sent to a@b\.c/);
	}
});

test('resume with approved=false emits tool-rejected and feeds rejection to model', async () => {
	let executed = false;
	const sendEmail = tool({
		needsApproval: true,
		inputSchema: z.object({ to: z.string() }),
		execute: () => {
			executed = true;
			return 'sent';
		},
	});

	const adapter = dummy({
		responses: [{ text: 'understood, not sending' }],
	});

	const chunks = await collect(
		chat({
			adapter,
			messages: [
				user('email a@b.c'),
				{
					role: 'assistant',
					content: [
						{
							type: 'tool-call',
							id: 'c1',
							name: 'sendEmail',
							arguments: JSON.stringify({ to: 'a@b.c' }),
							approval: { approved: false },
						},
					],
				},
			],
			tools: { sendEmail },
		}),
	);

	assert.equal(executed, false);

	const rejected = chunks.find((c) => c.type === 'tool-rejected');
	assert.ok(rejected);
	if (rejected?.type === 'tool-rejected' && rejected.name === 'sendEmail') {
		assert.equal(rejected.id, 'c1');
		assert.match(rejected.reason, /declined/i);
	}

	// the rejection landed in the conversation as a tool message before the model continues
	const lastMsg = adapter.calls[0]?.messages.at(-1);
	assert.equal(lastMsg?.role, 'tool');
	if (lastMsg?.role === 'tool') {
		const result = lastMsg.content[0];
		assert.equal(result?.toolCallId, 'c1');
		assert.match(result?.output ?? '', /declined/i);
		assert.equal(result?.isError, true);
	}

	const finish = chunks.at(-1);
	if (finish?.type === 'finish') {
		assert.equal(finish.reason, 'stop');
	}
});

test('mixed batch: a single gated call suspends every sibling', async () => {
	let safeRan = false;
	let gatedRan = false;
	const safe = tool({
		inputSchema: z.object({}),
		execute: () => {
			safeRan = true;
			return 'safe ok';
		},
	});
	const gated = tool({
		needsApproval: true,
		inputSchema: z.object({}),
		execute: () => {
			gatedRan = true;
			return 'gated ok';
		},
	});

	const adapter = dummy({
		responses: [
			{
				toolCalls: [
					{ id: 's1', name: 'safe', arguments: {} },
					{ id: 'g1', name: 'gated', arguments: {} },
					{ id: 's2', name: 'safe', arguments: {} },
				],
			},
		],
	});

	const chunks = await collect(
		chat({
			adapter,
			messages: [user('go')],
			tools: { safe, gated },
		}),
	);

	assert.equal(safeRan, false);
	assert.equal(gatedRan, false);

	// exactly one approval request, only for the gated call
	const approvals = chunks.filter((c) => c.type === 'tool-approval-requested');
	assert.equal(approvals.length, 1);
	if (approvals[0]?.type === 'tool-approval-requested') {
		assert.equal(approvals[0].id, 'g1');
	}

	const finish = chunks.at(-1);
	if (finish?.type === 'finish') {
		assert.equal(finish.reason, 'awaiting-approval');
	}
});

test('mixed batch resumes with all tools running once gated decision is given', async () => {
	const ran: string[] = [];
	const safe = tool({
		inputSchema: z.object({}),
		execute: (_, ctx) => {
			ran.push(ctx.toolCallId);
			return 'safe ok';
		},
	});
	const gated = tool({
		needsApproval: true,
		inputSchema: z.object({}),
		execute: (_, ctx) => {
			ran.push(ctx.toolCallId);
			return 'gated ok';
		},
	});

	const adapter = dummy({ responses: [{ text: 'all done' }] });

	await collect(
		chat({
			adapter,
			messages: [
				user('go'),
				{
					role: 'assistant',
					content: [
						{ type: 'tool-call', id: 's1', name: 'safe', arguments: '{}' },
						{
							type: 'tool-call',
							id: 'g1',
							name: 'gated',
							arguments: '{}',
							approval: { approved: true },
						},
						{ type: 'tool-call', id: 's2', name: 'safe', arguments: '{}' },
					],
				},
			],
			tools: { safe, gated },
		}),
	);

	// all three executed in order, only after approval
	assert.deepEqual(ran, ['s1', 'g1', 's2']);
});

test('emits message chunks for committed assistant and tool messages', async () => {
	const getWeather = tool({
		inputSchema: z.object({ city: z.string() }),
		execute: ({ city }) => ({ city, conditions: 'sunny' }),
	});

	const adapter = dummy({
		responses: [
			{ toolCalls: [{ id: 'c1', name: 'getWeather', arguments: { city: 'paris' } }] },
			{ text: 'sunny in paris.' },
		],
	});

	const chunks = await collect(
		chat({
			adapter,
			messages: [user('weather in paris?')],
			tools: { getWeather },
		}),
	);

	const messageChunks = chunks.filter((c) => c.type === 'message');
	// assistant turn 1 (with tool call), tool result, assistant turn 2 (text)
	assert.equal(messageChunks.length, 3);

	const m0 = messageChunks[0];
	if (m0?.type === 'message' && m0.message.role === 'assistant') {
		const callPart = m0.message.content.find((p): p is ToolCallPart => p.type === 'tool-call');
		assert.equal(callPart?.id, 'c1');
		assert.equal(callPart?.name, 'getWeather');
	} else {
		assert.fail('expected first message chunk to be an assistant turn with a tool call');
	}

	const m1 = messageChunks[1];
	if (m1?.type === 'message' && m1.message.role === 'tool') {
		const resultPart = m1.message.content[0];
		assert.equal(resultPart?.toolCallId, 'c1');
		assert.match(resultPart?.output ?? '', /sunny/);
	} else {
		assert.fail('expected second message chunk to be a tool result');
	}

	const m2 = messageChunks[2];
	if (m2?.type === 'message' && m2.message.role === 'assistant') {
		const textPart = m2.message.content[0];
		assert.equal(textPart?.type, 'text');
		if (textPart?.type === 'text') {
			assert.equal(textPart.text, 'sunny in paris.');
		}
	} else {
		assert.fail('expected third message chunk to be the final assistant turn');
	}
});

test('emits a message chunk for a synthetic rejection tool message', async () => {
	const sendEmail = tool({
		needsApproval: true,
		inputSchema: z.object({ to: z.string() }),
		execute: () => 'sent',
	});

	const adapter = dummy({ responses: [{ text: 'understood' }] });

	const chunks = await collect(
		chat({
			adapter,
			messages: [
				user('email a@b.c'),
				{
					role: 'assistant',
					content: [
						{
							type: 'tool-call',
							id: 'c1',
							name: 'sendEmail',
							arguments: JSON.stringify({ to: 'a@b.c' }),
							approval: { approved: false },
						},
					],
				},
			],
			tools: { sendEmail },
		}),
	);

	const rejectionMessage = chunks.find((c) => c.type === 'message' && c.message.role === 'tool');
	assert.ok(rejectionMessage, 'expected a tool message chunk for the rejection');
	if (rejectionMessage?.type === 'message' && rejectionMessage.message.role === 'tool') {
		const result = rejectionMessage.message.content[0];
		assert.equal(result?.toolCallId, 'c1');
		assert.match(result?.output ?? '', /declined/i);
	}
});

test('chained approval: resume approves, model emits a second gated call, suspends again', async () => {
	let firstRan = false;
	let secondRan = false;
	const sendEmail = tool({
		needsApproval: true,
		inputSchema: z.object({ to: z.string() }),
		execute: ({ to }) => {
			firstRan = true;
			return `sent to ${to}`;
		},
	});
	const deleteFile = tool({
		needsApproval: true,
		inputSchema: z.object({ path: z.string() }),
		execute: ({ path }) => {
			secondRan = true;
			return `deleted ${path}`;
		},
	});

	// after the first approved call executes, the model emits a *new* gated call
	const adapter = dummy({
		responses: [{ toolCalls: [{ id: 'd1', name: 'deleteFile', arguments: { path: '/tmp/x' } }] }],
	});

	const chunks = await collect(
		chat({
			adapter,
			messages: [
				user('do the thing'),
				{
					role: 'assistant',
					content: [
						{
							type: 'tool-call',
							id: 'e1',
							name: 'sendEmail',
							arguments: JSON.stringify({ to: 'a@b.c' }),
							approval: { approved: true },
						},
					],
				},
			],
			tools: { sendEmail, deleteFile },
		}),
	);

	// the approved call ran; the new gated call did not
	assert.equal(firstRan, true);
	assert.equal(secondRan, false);

	// the loop suspends a second time on the new gated call
	const finish = chunks.at(-1);
	assert.equal(finish?.type, 'finish');
	if (finish?.type === 'finish') {
		assert.equal(finish.reason, 'awaiting-approval');
	}

	const approvalReqs = chunks.filter((c) => c.type === 'tool-approval-requested');
	assert.equal(approvalReqs.length, 1);
	if (approvalReqs[0]?.type === 'tool-approval-requested') {
		assert.equal(approvalReqs[0].id, 'd1');
	}
});

test('partial approvals suspend: one decided + one pending → none execute', async () => {
	let firstRan = false;
	let secondRan = false;
	const sendEmail = tool({
		needsApproval: true,
		inputSchema: z.object({ to: z.string() }),
		execute: () => {
			firstRan = true;
			return 'sent';
		},
	});
	const deleteFile = tool({
		needsApproval: true,
		inputSchema: z.object({ path: z.string() }),
		execute: () => {
			secondRan = true;
			return 'deleted';
		},
	});

	const adapter = dummy({ responses: [{ text: 'unreachable' }] });

	const chunks = await collect(
		chat({
			adapter,
			messages: [
				user('go'),
				{
					role: 'assistant',
					content: [
						{
							type: 'tool-call',
							id: 'e1',
							name: 'sendEmail',
							arguments: JSON.stringify({ to: 'a@b.c' }),
							approval: { approved: true },
						},
						{
							type: 'tool-call',
							id: 'd1',
							name: 'deleteFile',
							arguments: JSON.stringify({ path: '/tmp/x' }),
						},
					],
				},
			],
			tools: { sendEmail, deleteFile },
		}),
	);

	// even though sendEmail was approved, deleteFile has no decision yet — nothing runs
	assert.equal(firstRan, false);
	assert.equal(secondRan, false);
	assert.equal(adapter.calls.length, 0);

	const finish = chunks.at(-1);
	if (finish?.type === 'finish') {
		assert.equal(finish.reason, 'awaiting-approval');
	}

	// only the undecided call surfaces a fresh approval request
	const approvalReqs = chunks.filter((c) => c.type === 'tool-approval-requested');
	assert.equal(approvalReqs.length, 1);
	if (approvalReqs[0]?.type === 'tool-approval-requested') {
		assert.equal(approvalReqs[0].id, 'd1');
	}
});

test('mixed approve/reject batch resumes once: approved executes, rejected injects message', async () => {
	const ran: string[] = [];
	const sendEmail = tool({
		needsApproval: true,
		inputSchema: z.object({ to: z.string() }),
		execute: ({ to }) => {
			ran.push('sendEmail');
			return `sent to ${to}`;
		},
	});
	const deleteFile = tool({
		needsApproval: true,
		inputSchema: z.object({ path: z.string() }),
		execute: () => {
			ran.push('deleteFile');
			return 'deleted';
		},
	});

	const adapter = dummy({ responses: [{ text: 'all done' }] });

	const chunks = await collect(
		chat({
			adapter,
			messages: [
				user('go'),
				{
					role: 'assistant',
					content: [
						{
							type: 'tool-call',
							id: 'e1',
							name: 'sendEmail',
							arguments: JSON.stringify({ to: 'a@b.c' }),
							approval: { approved: true },
						},
						{
							type: 'tool-call',
							id: 'd1',
							name: 'deleteFile',
							arguments: JSON.stringify({ path: '/tmp/x' }),
							approval: { approved: false },
						},
					],
				},
			],
			tools: { sendEmail, deleteFile },
		}),
	);

	// only the approved call ran
	assert.deepEqual(ran, ['sendEmail']);
	assert.equal(adapter.calls.length, 1);

	// resume detection skipped the model call for iteration 1; the model only
	// gets a turn after the batch finishes (one approved result + one rejection)
	const followUp = adapter.calls[0]?.messages ?? [];
	const tail = followUp.at(-1);
	assert.equal(tail?.role, 'tool');
	if (tail?.role === 'tool') {
		const results = tail.content as ToolResultPart[];
		assert.equal(results.length, 2);
		assert.equal(results[0]?.toolCallId, 'e1');
		assert.match(results[0]?.output ?? '', /sent to/);
		assert.equal(results[1]?.toolCallId, 'd1');
		assert.match(results[1]?.output ?? '', /declined/i);
	}

	const finish = chunks.at(-1);
	if (finish?.type === 'finish') {
		assert.equal(finish.reason, 'stop');
	}
});

test('tool registered under a different name than the tool definition', async () => {
	const weather = tool({
		inputSchema: z.object({ city: z.string() }),
		execute: ({ city }) => `weather for ${city}`,
	});

	const adapter = dummy({
		responses: [
			{ toolCalls: [{ id: 'c', name: 'get_weather_uk', arguments: { city: 'london' } }] },
			{ text: 'done' },
		],
	});

	const chunks = await collect(
		chat({
			adapter,
			messages: [user('go')],
			// same tool registered with different name
			tools: { get_weather_uk: weather, get_weather_us: weather },
		}),
	);

	const result = chunks.find((c) => c.type === 'tool-result');
	if (result?.type === 'tool-result') {
		assert.equal(result.name, 'get_weather_uk');
		assert.equal(result.result, 'weather for london');
	}
});

test('parallel tool calls produce a single tool message with batched results', async () => {
	const echo = tool({
		inputSchema: z.object({ s: z.string() }),
		execute: ({ s }) => s.toUpperCase(),
	});

	const adapter = dummy({
		responses: [
			{
				toolCalls: [
					{ id: 'c1', name: 'echo', arguments: { s: 'a' } },
					{ id: 'c2', name: 'echo', arguments: { s: 'b' } },
				],
			},
			{ text: 'done' },
		],
	});

	await collect(
		chat({
			adapter,
			messages: [user('go')],
			tools: { echo },
		}),
	);

	// the second turn sees one tool message with both results
	const secondTurn = adapter.calls[1];
	const lastMessage = secondTurn?.messages.at(-1);
	assert.equal(lastMessage?.role, 'tool');
	if (lastMessage?.role === 'tool') {
		assert.equal(lastMessage.content.length, 2);
		assert.equal(lastMessage.content[0]?.toolCallId, 'c1');
		assert.equal(lastMessage.content[0]?.output, 'A');
		assert.equal(lastMessage.content[1]?.toolCallId, 'c2');
		assert.equal(lastMessage.content[1]?.output, 'B');
	}
});
