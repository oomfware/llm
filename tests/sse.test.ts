import assert from 'node:assert/strict';
import { test } from 'node:test';

import { parseSseStream } from '../src/internal/sse.ts';

const streamOf = (chunks: string[]): ReadableStream<Uint8Array> => {
	const enc = new TextEncoder();
	return new ReadableStream({
		start(controller) {
			for (const c of chunks) {
				controller.enqueue(enc.encode(c));
			}
			controller.close();
		},
	});
};

const collect = async (stream: ReadableStream<Uint8Array>): Promise<string[]> => {
	const out: string[] = [];
	for await (const data of parseSseStream(stream)) {
		out.push(data);
	}
	return out;
};

test('parses single-chunk events', async () => {
	const events = await collect(streamOf(['data: hello\n\ndata: world\n\n']));
	assert.deepEqual(events, ['hello', 'world']);
});

test('reassembles events across binary chunk boundaries', async () => {
	const events = await collect(streamOf(['data: hel', 'lo\n\ndata:', ' world\n\n']));
	assert.deepEqual(events, ['hello', 'world']);
});

test('handles \\r\\n line endings', async () => {
	const events = await collect(streamOf(['data: a\r\n\r\ndata: b\r\n\r\n']));
	assert.deepEqual(events, ['a', 'b']);
});

test('skips comments and non-data fields', async () => {
	const events = await collect(streamOf([': comment\nevent: ping\ndata: payload\nid: 7\n\n']));
	assert.deepEqual(events, ['payload']);
});

test('dispatches trailing event data at stream end', async () => {
	const events = await collect(streamOf(['data: hello']));
	assert.deepEqual(events, ['hello']);
});

test('preserves [DONE] sentinel verbatim', async () => {
	const events = await collect(streamOf(['data: [DONE]\n\n']));
	assert.deepEqual(events, ['[DONE]']);
});
