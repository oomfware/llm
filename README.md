# @oomfware/llm

type-safe library for interacting with large language models.

```sh
npm install @oomfware/llm
```

the package talks to provider HTTP APIs directly through `fetch()`. no provider SDK is required at
runtime.

schemas use [Standard Schema](https://standardschema.dev/) plus
[Standard JSON Schema](https://standardschema.dev/json-schema), so tools and structured output work
with libraries such as zod, valibot, and arktype.

## usage

### choose a provider

```ts
import { anthropic, openai } from '@oomfware/llm';

const fast = openai('gpt-4o-mini', {
	apiKey: 'sk-...',
});
const reasoning = openai('gpt-5', {
	apiKey: 'sk-...',
});

const claude = anthropic('claude-sonnet-4-5', {
	apiKey: 'sk-ant-...',
});
```

provider factories do not read ambient configuration. pass `apiKey`, `baseUrl`, `headers`, or
`fetch` to configure the request layer.

### messages

every message is a discriminated union of role + ordered `content[]`. there is no `content: string`
shorthand — even simple text messages use a part array. helpers in `@oomfware/llm` keep the common
case short:

```ts
import { assistant, system, text, user } from '@oomfware/llm';

system('answer in one sentence.');
// → { role: 'system', content: [{ type: 'text', text: 'answer in one sentence.' }] }

user('what is sqlite?');
// → { role: 'user', content: [{ type: 'text', text: 'what is sqlite?' }] }
```

assistant turns interleave text, reasoning, and tool calls inside the same `content[]` in the order
the model produced them. this matters for prompt caching: anthropic and openai compute the cache
prefix against the wire bytes, so reordering parts between turns invalidates the cache.

### generate text

use `generate()` when you want one resolved result:

```ts
import { generate, openai, system, user } from '@oomfware/llm';

const result = await generate({
	adapter: openai('gpt-4o-mini'),
	messages: [system('answer in one sentence.'), user('what is sqlite?')],
});

console.log(result.text);
console.log(result.finishReason);
console.log(result.usage);
```

the result includes the accumulated assistant text, reasoning text, tool calls, tool results, usage,
finish reason, and the full conversation as `messages`.

### stream a chat

use `chat()` for streaming interfaces. it returns an async iterable of discriminated chunks.

```ts
import { chat, openai, user } from '@oomfware/llm';

const stream = chat({
	adapter: openai('gpt-4o-mini'),
	messages: [user('write a haiku about sqlite')],
});

for await (const chunk of stream) {
	if (chunk.type === 'text-delta') {
		process.stdout.write(chunk.delta);
	}
}
```

text arrives as `text-delta` chunks. providers that expose thinking content can also yield
`reasoning-delta`. each committed assistant or tool message is also surfaced as a `message` chunk —
append those to your history if you want to persist the conversation without rebuilding it from
deltas. the stream always ends with either `finish` or `error`.

### call tools

define tools with `tool()`, then register them by name on a chat call. the registration key is the
name sent to the model.

```ts
import { generate, openai, tool, user } from '@oomfware/llm';
import { z } from 'zod';

const getWeather = tool({
	description: 'get the current weather for a city',
	inputSchema: z.object({
		city: z.string(),
	}),
	execute: async ({ city }) => {
		return { city, conditions: 'sunny', temperatureC: 23 };
	},
});

const result = await generate({
	adapter: openai('gpt-4o-mini'),
	messages: [user('what is the weather in paris?')],
	tools: {
		get_weather: getWeather,
	},
});

console.log(result.text);
console.log(result.toolResults);
```

tool input is parsed from the model's JSON arguments, validated through the schema, and then passed
to `execute()`. add `outputSchema` when you also want to validate and transform the returned value
before it is yielded back into the loop.

```ts
const searchDocs = tool({
	inputSchema: z.object({
		query: z.string(),
	}),
	outputSchema: z.array(
		z.object({
			title: z.string(),
			url: z.string(),
		}),
	),
	execute: async ({ query }) => {
		return await search(query);
	},
});
```

tool chunks are typed against the registered tools. after narrowing on `chunk.name`, `input` and
`result` carry the right types:

```ts
for await (const chunk of chat({ adapter, messages, tools: { get_weather: getWeather } })) {
	if (chunk.type === 'tool-call-end' && chunk.name === 'get_weather') {
		console.log(chunk.input.city);
	}

	if (chunk.type === 'tool-result' && chunk.name === 'get_weather') {
		console.log(chunk.result.temperatureC);
	}
}
```

tools run sequentially. if you need concurrency, expose a batching tool and run the inner work in
parallel inside `execute()`. when the model emits multiple tool calls in one turn, all results are
batched into a single `ToolMessage` whose `content` is an array of `tool-result` parts.

### require approval

set `needsApproval: true` for tools that should pause before side effects. the loop suspends with
`finishReason: 'awaiting-approval'` and leaves the pending tool calls inside the assistant turn's
`content[]` as `tool-call` parts.

```ts
import { generate, openai, tool, user } from '@oomfware/llm';
import { z } from 'zod';

const sendEmail = tool({
	needsApproval: true,
	description: 'send an email',
	inputSchema: z.object({
		to: z.string(),
		body: z.string(),
	}),
	execute: async ({ to, body }) => {
		await sendgrid.send({ to, body });
		return { sent: true };
	},
});

const first = await generate({
	adapter: openai('gpt-4o-mini'),
	messages: [user('email alice the meeting notes')],
	tools: { sendEmail },
});

if (first.finishReason === 'awaiting-approval') {
	const tail = first.messages.at(-1);

	if (tail?.role === 'assistant') {
		for (const part of tail.content) {
			if (part.type === 'tool-call') {
				part.approval = { approved: await approve(part) };
			}
		}
	}

	const second = await generate({
		adapter: openai('gpt-4o-mini'),
		messages: first.messages,
		tools: { sendEmail },
	});

	console.log(second.text);
}
```

approvals live on the persisted `tool-call` parts. approved calls execute on resume without asking
the model to recreate them. rejected calls produce a synthetic `tool-result` with `isError: true`
and a "user declined" message, then the model gets another turn.

if one tool call in a batch needs approval, the whole batch waits. non-gated sibling calls are not
executed speculatively.

with `chat()`, the same pattern works against the stream. consume `message` chunks to keep your
history in sync, watch for `tool-approval-requested` to know what to prompt for, and check the
finish reason to decide whether to resume:

```ts
import { chat, openai, user } from '@oomfware/llm';
import type { ModelMessage } from '@oomfware/llm';

const messages: ModelMessage[] = [user('email alice the meeting notes')];

while (true) {
	let finishReason: string | undefined;

	for await (const chunk of chat({
		adapter: openai('gpt-4o-mini'),
		messages,
		tools: { sendEmail },
	})) {
		switch (chunk.type) {
			case 'text-delta':
				process.stdout.write(chunk.delta);
				break;
			case 'tool-approval-requested':
				console.log(`approve ${chunk.name}?`, chunk.input);
				break;
			case 'message':
				messages.push(chunk.message);
				break;
			case 'finish':
				finishReason = chunk.reason;
				break;
		}
	}

	if (finishReason !== 'awaiting-approval') break;

	const tail = messages.at(-1);
	if (tail?.role === 'assistant') {
		for (const part of tail.content) {
			if (part.type === 'tool-call' && !part.approval) {
				part.approval = { approved: await approve(part) };
			}
		}
	}
}
```

### return structured data

`generateObject()` runs the normal agent loop first, then makes one final structured-output call
against the same adapter.

```ts
import { generateObject, openai, user } from '@oomfware/llm';
import { z } from 'zod';

const result = await generateObject({
	adapter: openai('gpt-4o-mini'),
	messages: [user('extract the event from: dinner tomorrow at 7pm')],
	schemaName: 'event',
	schema: z.object({
		title: z.string(),
		when: z.string(),
	}),
});

console.log(result.object.title);
console.log(result.object.when);
```

the returned object is typed from the schema and validated before it is returned. tools can be used
during the loop before the final structured response:

```ts
const lookupCalendar = tool({
	inputSchema: z.object({ name: z.string() }),
	execute: async ({ name }) => {
		return await findCalendar(name);
	},
});

const result = await generateObject({
	adapter: openai('gpt-4o-mini'),
	messages: [user('which calendar should dinner go on?')],
	tools: { lookup_calendar: lookupCalendar },
	schema: z.object({
		calendarId: z.string(),
		reason: z.string(),
	}),
});
```

OpenAI uses strict JSON Schema response format. Anthropic is adapted through a forced tool-style
structured response.

### use plain JSON Schema

use `jsonSchema()` when your schema already exists as JSON Schema instead of a Standard Schema
library value.

```ts
import { jsonSchema, tool } from '@oomfware/llm';

const schema = jsonSchema<{ city: string }>(
	{
		type: 'object',
		properties: {
			city: { type: 'string' },
		},
		required: ['city'],
	},
	(raw) => {
		if (typeof raw === 'object' && raw !== null && 'city' in raw && typeof raw.city === 'string') {
			return { city: raw.city };
		}

		throw new Error('city is required');
	},
);

const getWeather = tool({
	inputSchema: schema,
	execute: ({ city }) => `weather for ${city}`,
});
```

without a parser, `jsonSchema<T>()` trusts the type parameter and passes parsed JSON through
unchanged.

## prompt caching

every message and every part has an optional `providerMetadata` bag — a `Record<string, unknown>`
keyed by provider name. adapters use it to round-trip provider-specific opaque state and to control
provider-specific features without polluting the core types.

### anthropic cache breakpoints

attach `providerMetadata.anthropic.cacheControl` to the part you want to mark as a cache breakpoint:

```ts
import { chat, anthropic, user } from '@oomfware/llm';

chat({
	adapter: anthropic('claude-sonnet-4-5'),
	messages: [
		{
			role: 'user',
			content: [
				{ type: 'text', text: longContextDocument },
				{
					type: 'text',
					text: 'now answer this question:',
					providerMetadata: { anthropic: { cacheControl: { type: 'ephemeral' } } },
				},
				{ type: 'text', text: question },
			],
		},
	],
});
```

attaching `cacheControl` at the message level stamps the marker on the last block of that message —
a convenient shorthand when you don't care about the exact part:

```ts
user('a really long document...', {
	providerMetadata: { anthropic: { cacheControl: { type: 'ephemeral', ttl: '1h' } } },
});
```

usage reported by the adapter exposes cache hits and writes:

```ts
const result = await generate({ adapter: anthropic('claude-sonnet-4-5'), messages });
console.log(result.usage?.cacheReadInputTokens); // anthropic prompt-cache hit
console.log(result.usage?.cacheCreationInputTokens); // anthropic prompt-cache write
```

### extended thinking

when you enable extended thinking, anthropic returns thinking blocks with opaque `signature` fields.
the adapter captures them automatically and replays them verbatim on the next turn, so prompt
caching still works across resumes:

```ts
import { generate, anthropic, user } from '@oomfware/llm';

const result = await generate({
	adapter: anthropic('claude-sonnet-4-5'),
	messages: [user('think hard about this problem')],
	providerOptions: { thinking: { type: 'enabled', budgetTokens: 4096 } },
});
```

the persisted assistant message will contain `reasoning` and `redacted-reasoning` parts with the
required signatures stashed in `providerMetadata.anthropic`. you don't have to manage them yourself
— just keep the message around.

### openai reasoning models

reasoning models return `reasoning` items that must round-trip on resume. the adapter captures their
`id`, `encryptedContent`, `summary`, `content`, and `status` automatically. it also requests
`include: ['reasoning.encrypted_content']` whenever `reasoningEffort` is set, so encrypted reasoning
flows through stateless conversations correctly.

```ts
import { generate, openai, user } from '@oomfware/llm';

const result = await generate({
	adapter: openai('gpt-5'),
	messages: [user('reason carefully')],
	providerOptions: { reasoningEffort: 'high' },
});

console.log(result.usage?.reasoningTokens);
```

set `includeEncryptedReasoning: false` to opt out if you don't need stateless replay.

## loop control

the agent loop stops when the model stops asking for tools. it is also capped at 5 model turns by
default.

```ts
import {
	chat,
	combineStrategies,
	maxIterations,
	openai,
	untilFinishReason,
	user,
} from '@oomfware/llm';

const stream = chat({
	adapter: openai('gpt-4o-mini'),
	messages: [user('keep trying until you are filtered')],
	agentLoopStrategy: combineStrategies(maxIterations(10), untilFinishReason('content-filter')),
});
```

a custom strategy is a function that receives the current loop state and returns whether the loop
should continue.

```ts
import type { AgentLoopStrategy } from '@oomfware/llm';

const stayCheap: AgentLoopStrategy = (state) => {
	return state.totalUsage === undefined || state.totalUsage.outputTokens < 1_000;
};
```

## provider options

provider options are typed from the selected adapter. model-specific OpenAI options only appear for
models that support them.

```ts
import { chat, openai, user } from '@oomfware/llm';

const adapter = openai('gpt-5');

chat({
	adapter,
	messages: [user('think carefully')],
	providerOptions: {
		reasoningEffort: 'high',
	},
});
```

Anthropic options include `topK`, `stopSequences`, and `thinking`.

## custom adapters

adapters implement streaming chat plus structured output. use `createChatAdapter()` when adding a
provider.

```ts
import { createChatAdapter, type ChatAdapter } from '@oomfware/llm';

const local = (model: string): ChatAdapter<string, { numCtx?: number }> => {
	return createChatAdapter({
		kind: 'chat',
		provider: 'local',
		model,
		async *chatStream(options) {
			yield { type: 'text-start', id: 't0' };
			yield { type: 'text-delta', id: 't0', delta: await callLocalModel(options) };
			yield { type: 'text-end', id: 't0' };
			yield { type: 'finish', reason: 'stop' };
		},
		async structuredOutput(options) {
			const data = await callLocalJsonModel(options);
			return { data, rawText: JSON.stringify(data) };
		},
	});
};
```

adapters emit text and reasoning parts as `*-start` / `*-delta` / `*-end` events keyed by an id of
the adapter's choosing, plus `tool-call-start` / `tool-call-delta` / `tool-call-end` for tool calls.
the agent loop accumulates parts in receive order to build the assistant turn's `content[]`.

for tests, use `dummy()` to script model turns:

```ts
import { dummy } from '@oomfware/llm';

const adapter = dummy({
	responses: [
		{
			toolCalls: [{ id: 'call_1', name: 'get_weather', arguments: { city: 'paris' } }],
		},
		{ text: 'it is sunny in paris.' },
	],
	structuredOutputs: [{ city: 'paris', conditions: 'sunny' }],
});
```
