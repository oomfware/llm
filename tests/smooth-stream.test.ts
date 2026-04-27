import assert from 'node:assert/strict';
import { test } from 'node:test';

import { chat } from '../src/chat.ts';
import { user } from '../src/messages.ts';
import { dummy } from '../src/providers/dummy.ts';
import { smoothStream } from '../src/smooth-stream.ts';
import type { StreamChunk } from '../src/types.ts';

const collect = async <T extends StreamChunk<any>>(stream: AsyncIterable<T>): Promise<T[]> => {
	const out: T[] = [];
	for await (const chunk of stream) {
		out.push(chunk);
	}
	return out;
};

const makeStream = async function* (chunks: StreamChunk[]): AsyncGenerator<StreamChunk, void, void> {
	for (const c of chunks) {
		yield c;
	}
};

test('word chunking emits at whitespace boundaries and flushes the tail', async () => {
	const input: StreamChunk[] = [
		{ type: 'text-delta', delta: 'hel' },
		{ type: 'text-delta', delta: 'lo wor' },
		{ type: 'text-delta', delta: 'ld and' },
		{ type: 'text-delta', delta: ' bye' },
		{ type: 'finish', reason: 'stop' },
	];

	const out = await collect(smoothStream(makeStream(input), { chunking: 'word' }));
	const texts = out.filter((c) => c.type === 'text-delta').map((c) => c.delta);

	assert.deepEqual(texts, ['hello ', 'world ', 'and ', 'bye']);
	assert.equal(out.at(-1)?.type, 'finish');
});

test('line chunking emits per newline, flushes remainder before finish', async () => {
	const input: StreamChunk[] = [
		{ type: 'text-delta', delta: 'one\ntw' },
		{ type: 'text-delta', delta: 'o\nthree' },
		{ type: 'finish', reason: 'stop' },
	];

	const out = await collect(smoothStream(makeStream(input), { chunking: 'line' }));
	const texts = out.filter((c) => c.type === 'text-delta').map((c) => c.delta);

	assert.deepEqual(texts, ['one\n', 'two\n', 'three']);
});

test('regex chunking is supported and global flag is normalised', async () => {
	const input: StreamChunk[] = [
		{ type: 'text-delta', delta: 'a,b,' },
		{ type: 'text-delta', delta: 'c,d' },
		{ type: 'finish', reason: 'stop' },
	];

	const out = await collect(smoothStream(makeStream(input), { chunking: /[^,]*,/g }));
	const texts = out.filter((c) => c.type === 'text-delta').map((c) => c.delta);

	assert.deepEqual(texts, ['a,', 'b,', 'c,', 'd']);
});

test('non-text-delta chunks pass through and pending text flushes before them', async () => {
	const input: StreamChunk[] = [
		{ type: 'text-delta', delta: 'partial' },
		{ type: 'reasoning-delta', delta: 'thinking' },
		{ type: 'text-delta', delta: 'after ' },
		{ type: 'finish', reason: 'stop' },
	];

	const out = await collect(smoothStream(makeStream(input), { chunking: 'word' }));

	// 'partial' has no trailing whitespace, so it must be flushed on the
	// reasoning-delta boundary; 'after ' has a space and emits naturally.
	const types = out.map((c) => c.type);
	assert.deepEqual(types, ['text-delta', 'reasoning-delta', 'text-delta', 'finish']);

	const deltas = out.filter((c) => c.type === 'text-delta').map((c) => c.delta);
	assert.deepEqual(deltas, ['partial', 'after ']);
});

test('preserves typed tool chunks through the transform', async () => {
	const adapter = dummy({ responses: [{ text: 'hello world bye' }] });

	const out = await collect(
		smoothStream(
			chat({
				adapter,
				messages: [user('hi')],
			}),
			{ chunking: 'word' },
		),
	);

	const text = out
		.filter((c) => c.type === 'text-delta')
		.map((c) => c.delta)
		.join('');
	assert.equal(text, 'hello world bye');
	assert.equal(out.at(-1)?.type, 'finish');
});

test('delayMs paces emission without dropping output', async () => {
	const input: StreamChunk[] = [
		{ type: 'text-delta', delta: 'a b c ' },
		{ type: 'finish', reason: 'stop' },
	];

	const start = Date.now();
	const out = await collect(smoothStream(makeStream(input), { chunking: 'word', delayMs: 5 }));
	const elapsed = Date.now() - start;

	const deltas = out.filter((c) => c.type === 'text-delta').map((c) => c.delta);
	assert.deepEqual(deltas, ['a ', 'b ', 'c ']);
	// 3 emits @ 5ms each — generous lower bound to avoid flakiness on slow ci.
	assert.ok(elapsed >= 10, `expected >= 10ms, got ${elapsed}ms`);
});
