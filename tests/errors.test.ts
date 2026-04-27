import assert from 'node:assert/strict';
import { test } from 'node:test';

import { z } from 'zod';

import { chat } from '../src/chat.ts';
import {
	AIError,
	StructuredOutputValidationError,
	ToolArgumentsParseError,
	ToolInputValidationError,
	ToolOutputValidationError,
	UnknownToolError,
} from '../src/errors.ts';
import { generateObject } from '../src/generate-object.ts';
import { user } from '../src/messages.ts';
import { dummy } from '../src/providers/dummy.ts';
import { tool } from '../src/tool.ts';
import type { StreamChunk } from '../src/types.ts';

const collect = async <T extends StreamChunk<any>>(stream: AsyncIterable<T>): Promise<T[]> => {
	const out: T[] = [];
	for await (const chunk of stream) {
		out.push(chunk);
	}
	return out;
};

test('UnknownToolError carries name + id and is an AIError', async () => {
	const adapter = dummy({
		responses: [{ toolCalls: [{ id: 'c_1', name: 'nope', arguments: {} }] }],
	});

	const chunks = await collect(
		chat({
			adapter,
			messages: [user('go')],
		}),
	);

	const err = chunks.find((c) => c.type === 'error');
	assert.ok(err && err.type === 'error');
	assert.ok(err.error instanceof UnknownToolError);
	assert.ok(err.error instanceof AIError);
	if (err.error instanceof UnknownToolError) {
		assert.equal(err.error.name, 'UnknownToolError');
		assert.equal(err.error.toolName, 'nope');
		assert.equal(err.error.toolCallId, 'c_1');
	}
});

test('ToolInputValidationError carries issues and raw arguments', async () => {
	const strict = tool({
		inputSchema: z.object({ count: z.number().int().positive() }),
		execute: ({ count }) => count,
	});

	const adapter = dummy({
		responses: [{ toolCalls: [{ id: 'c_1', name: 'strict', arguments: { count: -1 } }] }],
	});

	const chunks = await collect(
		chat({
			adapter,
			messages: [user('go')],
			tools: { strict },
		}),
	);

	const err = chunks.find((c) => c.type === 'error');
	assert.ok(err && err.type === 'error');
	assert.ok(err.error instanceof ToolInputValidationError);
	if (err.error instanceof ToolInputValidationError) {
		assert.equal(err.error.toolName, 'strict');
		assert.equal(err.error.toolCallId, 'c_1');
		assert.ok(err.error.issues.length > 0);
		assert.equal(err.error.rawArguments, JSON.stringify({ count: -1 }));
	}
});

test('ToolOutputValidationError carries the raw bad return value', async () => {
	const misbehaving = tool({
		inputSchema: z.object({}),
		outputSchema: z.object({ value: z.number() }),
		execute: () => ({ value: 'not a number' }) as unknown as { value: number },
	});

	const adapter = dummy({
		responses: [{ toolCalls: [{ id: 'c_1', name: 'misbehaving', arguments: {} }] }],
	});

	const chunks = await collect(
		chat({
			adapter,
			messages: [user('go')],
			tools: { misbehaving },
		}),
	);

	const err = chunks.find((c) => c.type === 'error');
	assert.ok(err && err.type === 'error');
	assert.ok(err.error instanceof ToolOutputValidationError);
	if (err.error instanceof ToolOutputValidationError) {
		assert.equal(err.error.toolName, 'misbehaving');
		assert.equal(err.error.toolCallId, 'c_1');
		assert.deepEqual(err.error.rawResult, { value: 'not a number' });
	}
});

test('ToolArgumentsParseError preserves the parse cause and raw string', async () => {
	const t = tool({
		inputSchema: z.object({}),
		execute: () => 'ok',
	});

	// dummy provider stringifies arguments; bypass it by feeding raw chunks
	// through a custom adapter so we can inject malformed json.
	const adapter = {
		kind: 'chat' as const,
		provider: 'malformed',
		model: 'm',
		'~types': { providerOptions: undefined as unknown },
		// oxlint-disable-next-line typescript/require-await
		async *chatStream() {
			yield { type: 'tool-call-start' as const, name: 't', id: 'c_1' };
			yield { type: 'tool-call-delta' as const, name: 't', id: 'c_1', argsDelta: '{not-json' };
			yield { type: 'tool-call-end' as const, name: 't', id: 'c_1' };
			yield { type: 'finish' as const, reason: 'tool-calls' as const };
		},
		async structuredOutput() {
			return { data: {}, rawText: '' };
		},
	};

	const chunks = await collect(
		chat({
			adapter,
			messages: [user('go')],
			tools: { t },
		}),
	);

	const err = chunks.find((c) => c.type === 'error');
	assert.ok(err && err.type === 'error');
	assert.ok(err.error instanceof ToolArgumentsParseError);
	if (err.error instanceof ToolArgumentsParseError) {
		assert.equal(err.error.toolName, 't');
		assert.equal(err.error.toolCallId, 'c_1');
		assert.equal(err.error.rawArguments, '{not-json');
		assert.ok(err.error.cause instanceof SyntaxError);
	}
});

test('StructuredOutputValidationError surfaces issues + raw data', async () => {
	const adapter = dummy({
		responses: [{ text: 'ok' }],
		structuredOutputs: [{ width: 'wide', height: 'tall' }],
	});

	await assert.rejects(
		generateObject({
			adapter,
			messages: [user('how big?')],
			schema: z.object({ width: z.number(), height: z.number() }),
		}),
		(err) => {
			assert.ok(err instanceof StructuredOutputValidationError);
			assert.ok(err instanceof AIError);
			if (err instanceof StructuredOutputValidationError) {
				assert.deepEqual(err.rawData, { width: 'wide', height: 'tall' });
				assert.ok(err.issues.length > 0);
			}
			return true;
		},
	);
});
