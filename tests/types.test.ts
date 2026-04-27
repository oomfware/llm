/**
 * type-only checks. these tests are no-ops at runtime; the `@ts-expect-error`
 * directives prove that:
 *   - per-model providerOptions narrow correctly
 *   - tool chunks discriminate on `name` to narrow `input` and `result`
 *
 * each test wraps assertions in a never-called function so the body is type
 * checked but performs no I/O.
 */
import { test } from 'node:test';

import { z } from 'zod';

import { chat } from '../src/chat.ts';
import { user } from '../src/messages.ts';
import { anthropic } from '../src/providers/anthropic.ts';
import { openai } from '../src/providers/openai.ts';
import { tool, type InferToolInput, type InferToolOutput } from '../src/tool.ts';
import type { StreamChunk } from '../src/types.ts';

test('per-model providerOptions narrowing (type-only)', () => {
	const _check = () => {
		const reasoner = openai('gpt-5', { apiKey: 'x' });
		type ReasonerOptions = Parameters<typeof chat<typeof reasoner>>[0];
		const _r1: ReasonerOptions = {
			adapter: reasoner,
			messages: [user('hi')],
			providerOptions: { reasoningEffort: 'high' },
		};
		const _r2: ReasonerOptions = {
			adapter: reasoner,
			messages: [user('hi')],
			// @ts-expect-error - 'extreme' is not a valid reasoning effort
			providerOptions: { reasoningEffort: 'extreme' },
		};

		const plain = openai('gpt-4o', { apiKey: 'x' });
		type PlainOptions = Parameters<typeof chat<typeof plain>>[0];
		const _p: PlainOptions = {
			adapter: plain,
			messages: [user('hi')],
			// @ts-expect-error - reasoningEffort is not on gpt-4o's options
			providerOptions: { reasoningEffort: 'high' },
		};

		const claude = anthropic('claude-sonnet-4-5', { apiKey: 'x' });
		type ClaudeOptions = Parameters<typeof chat<typeof claude>>[0];
		const _c1: ClaudeOptions = {
			adapter: claude,
			messages: [user('hi')],
			providerOptions: { topK: 40, thinking: { type: 'enabled', budgetTokens: 4096 } },
		};
		const _c2: ClaudeOptions = {
			adapter: claude,
			messages: [user('hi')],
			// @ts-expect-error - reasoningEffort is openai-only
			providerOptions: { reasoningEffort: 'high' },
		};

		void _r1;
		void _r2;
		void _p;
		void _c1;
		void _c2;
	};
	void _check;
});

test('tool chunk discrimination by name narrows input and result (type-only)', () => {
	const getWeather = tool({
		inputSchema: z.object({ city: z.string() }),
		execute: ({ city }) => ({ city, conditions: 'sunny' as const }),
	});

	const searchWeb = tool({
		inputSchema: z.object({ query: z.string(), limit: z.number().optional() }),
		execute: ({ query }) => ({ results: [`hit for ${query}`] }),
	});

	type Tools = { get_weather: typeof getWeather; search_web: typeof searchWeb };

	const _check = (chunk: StreamChunk<Tools>) => {
		if (chunk.type === 'tool-call-end') {
			const _n: 'get_weather' | 'search_web' = chunk.name;
			void _n;

			if (chunk.name === 'get_weather') {
				const _city: string = chunk.input.city;
				void _city;
				// @ts-expect-error - 'query' is not on getWeather's input
				void chunk.input.query;
			}

			if (chunk.name === 'search_web') {
				const _query: string = chunk.input.query;
				void _query;
				// @ts-expect-error - 'city' is not on searchWeb's input
				void chunk.input.city;
			}
		}

		if (chunk.type === 'tool-result') {
			if (chunk.name === 'get_weather') {
				const _conditions: 'sunny' = chunk.result.conditions;
				void _conditions;
				// @ts-expect-error - 'results' is not on getWeather's output
				void chunk.result.results;
			}

			if (chunk.name === 'search_web') {
				const _results: string[] = chunk.result.results;
				void _results;
			}
		}
	};
	void _check;
});

test('tools registered under different names than the tool definition (type-only)', () => {
	const weather = tool({
		inputSchema: z.object({ city: z.string() }),
		execute: ({ city }) => `weather for ${city}`,
	});

	type Tools = { get_weather_uk: typeof weather; get_weather_us: typeof weather };

	const _check = (chunk: StreamChunk<Tools>) => {
		if (chunk.type === 'tool-call-start') {
			const _n: 'get_weather_uk' | 'get_weather_us' = chunk.name;
			void _n;
		}
	};
	void _check;
});

test('outputSchema makes InferToolOutput follow the schema (type-only)', () => {
	const fetchTodos = tool({
		inputSchema: z.object({ userId: z.string() }),
		outputSchema: z.array(z.object({ id: z.string(), title: z.string() })),
		execute: async ({ userId }) => [{ id: userId, title: 'todo' }],
	});

	type TodoIn = InferToolInput<typeof fetchTodos>;
	type TodoOut = InferToolOutput<typeof fetchTodos>;
	const _in: TodoIn = { userId: 'u1' };
	const _out: TodoOut = [{ id: '1', title: 't' }];
	void _in;
	void _out;

	// @ts-expect-error - output element shape is fixed by the schema
	const _bad: TodoOut = [{ id: '1' }];
	void _bad;
});

test('without outputSchema, output infers from execute return (type-only)', () => {
	const echo = tool({
		inputSchema: z.object({ s: z.string() }),
		execute: ({ s }) => ({ uppercased: s.toUpperCase() }),
	});

	type EchoOut = InferToolOutput<typeof echo>;
	const _out: EchoOut = { uppercased: 'HI' };
	void _out;

	// @ts-expect-error - execute returns { uppercased: string }, not { wrong: ... }
	const _bad: EchoOut = { wrong: 'HI' };
	void _bad;
});

test('chat without tools: stream chunk union excludes tool variants (type-only)', () => {
	const _check = (chunk: StreamChunk) => {
		if (chunk.type === 'text-delta') {
			const _s: string = chunk.delta;
			void _s;
		}
		// @ts-expect-error - tool-call-start should not be in the union when tools is omitted
		if (chunk.type === 'tool-call-start') {
			void chunk;
		}
	};
	void _check;
});
