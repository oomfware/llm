import assert from 'node:assert/strict';
import { test } from 'node:test';

import { z } from 'zod';

import { generate } from '../src/generate.ts';
import { user } from '../src/messages.ts';
import { dummy } from '../src/providers/dummy.ts';
import { tool } from '../src/tool.ts';

test('returns concatenated text + finish + usage', async () => {
	const adapter = dummy({ responses: [{ text: 'hello world' }] });
	const result = await generate({
		adapter,
		messages: [user('hi')],
	});

	assert.equal(result.text, 'hello world');
	assert.equal(result.finishReason, 'stop');
	assert.equal(result.toolCalls.length, 0);
	assert.equal(result.toolResults.length, 0);
	// input + final assistant turn
	assert.equal(result.messages.length, 2);
	assert.equal(result.messages[1]?.role, 'assistant');
});

test('captures tool calls and results across iterations', async () => {
	const getWeather = tool({
		inputSchema: z.object({ city: z.string() }),
		execute: ({ city }) => ({ city, temp: 72 }),
	});

	const adapter = dummy({
		responses: [
			{
				text: 'looking up weather...',
				toolCalls: [{ id: 'call_1', name: 'get_weather', arguments: { city: 'paris' } }],
			},
			{ text: 'it is 72 in paris.' },
		],
	});

	const result = await generate({
		adapter,
		messages: [user('weather?')],
		tools: { get_weather: getWeather },
	});

	assert.equal(result.text, 'looking up weather...it is 72 in paris.');
	assert.equal(result.toolCalls.length, 1);
	assert.equal(result.toolCalls[0]?.name, 'get_weather');
	assert.equal(result.toolResults.length, 1);
	assert.deepEqual(result.toolResults[0]?.result, { city: 'paris', temp: 72 });

	// input + assistant(text+toolCalls) + tool(result) + assistant(text)
	assert.equal(result.messages.length, 4);
	assert.equal(result.messages[1]?.role, 'assistant');
	assert.equal(result.messages[2]?.role, 'tool');
	assert.equal(result.messages[3]?.role, 'assistant');
});

test('captures reasoning content separately from text', async () => {
	const adapter = dummy({
		responses: [
			{
				parts: [
					{ type: 'reasoning', text: 'pondering…' },
					{ type: 'text', text: 'answer' },
				],
			},
		],
	});
	const result = await generate({
		adapter,
		messages: [user('hi')],
	});
	assert.equal(result.reasoning, 'pondering…');
	assert.equal(result.text, 'answer');
});

test('throws on adapter error', async () => {
	const adapter = dummy({ responses: [{ error: new Error('boom') }] });
	await assert.rejects(() => generate({ adapter, messages: [user('hi')] }), /boom/);
});

test('approval: suspended turn surfaces toolCalls and awaiting-approval; messages can resume', async () => {
	const sendEmail = tool({
		needsApproval: true,
		inputSchema: z.object({ to: z.string() }),
		execute: ({ to }) => `sent to ${to}`,
	});

	const adapter = dummy({
		responses: [{ toolCalls: [{ id: 'c1', name: 'sendEmail', arguments: { to: 'a@b.c' } }] }],
	});

	const first = await generate({
		adapter,
		messages: [user('email a@b.c')],
		tools: { sendEmail },
	});

	assert.equal(first.finishReason, 'awaiting-approval');
	assert.equal(first.toolCalls.length, 1);
	assert.equal(first.toolCalls[0]?.id, 'c1');

	// the assistant turn carrying the gated call must be in messages so a
	// subsequent generate() call can detect it as a resumption point.
	const lastMsg = first.messages.at(-1);
	assert.equal(lastMsg?.role, 'assistant');
	if (lastMsg?.role === 'assistant') {
		const callPart = lastMsg.content.find((p) => p.type === 'tool-call');
		assert.equal(callPart?.id, 'c1');
	}
});

test('approval: rejection lands as a tool message in result.messages', async () => {
	const sendEmail = tool({
		needsApproval: true,
		inputSchema: z.object({ to: z.string() }),
		execute: () => 'sent',
	});

	const adapter = dummy({ responses: [{ text: 'ok, not sending' }] });

	const result = await generate({
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
	});

	assert.equal(result.finishReason, 'stop');

	// rejection synthetic tool message is present and the model's reply follows it
	const toolMsg = result.messages.find((m) => m.role === 'tool');
	assert.ok(toolMsg, 'expected a tool message for the rejection');
	if (toolMsg?.role === 'tool') {
		const part = toolMsg.content[0];
		assert.equal(part?.toolCallId, 'c1');
		assert.match(part?.output ?? '', /declined/i);
	}
});
