import assert from 'node:assert/strict';
import { test } from 'node:test';

import { assistant, system, text, toolResults, user } from '../src/messages.ts';

test('text: builds a TextPart from a string', () => {
	assert.deepEqual(text('hello'), { type: 'text', text: 'hello', providerMetadata: undefined });
});

test('text: passes through providerMetadata when supplied', () => {
	const meta = { anthropic: { cacheControl: { type: 'ephemeral' } } };
	assert.deepEqual(text('cached', { providerMetadata: meta }), {
		type: 'text',
		text: 'cached',
		providerMetadata: meta,
	});
});

test('system: wraps a string in a SystemMessage with a single text part', () => {
	assert.deepEqual(system('be terse.'), {
		role: 'system',
		content: [{ type: 'text', text: 'be terse.' }],
		providerMetadata: undefined,
	});
});

test('user: accepts a plain string and wraps it as a single text part', () => {
	assert.deepEqual(user('hi'), {
		role: 'user',
		content: [{ type: 'text', text: 'hi' }],
		providerMetadata: undefined,
	});
});

test('user: accepts a TextPart array verbatim', () => {
	const parts = [
		text('a'),
		text('b', { providerMetadata: { anthropic: { cacheControl: { type: 'ephemeral' } } } }),
	];
	const message = user(parts);
	assert.equal(message.role, 'user');
	assert.equal(message.content, parts);
});

test('user: passes through providerMetadata at the message level', () => {
	const meta = { openai: { id: 'msg_1' } };
	const message = user('hi', { providerMetadata: meta });
	assert.equal(message.providerMetadata, meta);
});

test('assistant: accepts a plain string', () => {
	assert.deepEqual(assistant('done'), {
		role: 'assistant',
		content: [{ type: 'text', text: 'done' }],
		providerMetadata: undefined,
	});
});

test('assistant: accepts AssistantContent with reasoning + tool-call parts', () => {
	const content = [
		{ type: 'reasoning' as const, text: 'thinking…' },
		{ type: 'tool-call' as const, id: 'c1', name: 'getWeather', arguments: '{"city":"paris"}' },
	];
	const message = assistant(content);
	assert.equal(message.content, content);
});

test('toolResults: wraps results into a ToolMessage', () => {
	const results = [{ type: 'tool-result' as const, toolCallId: 'c1', output: '{"ok":true}' }];
	const message = toolResults(results);
	assert.deepEqual(message, { role: 'tool', content: results, providerMetadata: undefined });
});
