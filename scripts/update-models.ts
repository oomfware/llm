// fetches openrouter's public catalog and regenerates the literal-union files
// under `src/providers/generated/`. run via `pnpm update-models`.
//
// openrouter is the single source of truth here — its catalog covers all
// providers we care about and requires no api key. for anthropic we strip
// the `anthropic/` prefix and normalize dots to dashes (so openrouter's
// `claude-sonnet-4.5` becomes anthropic's native `claude-sonnet-4-5`).
//
// pass `--only=anthropic,openai,openrouter` to limit which providers run.

import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { z } from 'zod';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const OUTPUT_DIR = resolve(ROOT, 'src/providers/generated');
const OPENROUTER_URL = 'https://openrouter.ai/api/v1/models';
const OPENROUTER_RANKED_URL = 'https://openrouter.ai/api/frontend/models/find?order=top-weekly';

/**
 * cap the openrouter union to the N most-used models. anthropic/openai unions stay full because their slugs
 * are pre-filtered by prefix and already small.
 */
const OPENROUTER_TOP_N = 150;

const openRouterModelSchema = z.object({
	id: z.string(),
	architecture: z
		.object({
			input_modalities: z.array(z.string()).optional(),
			output_modalities: z.array(z.string()).optional(),
		})
		.optional(),
	/** ISO date (YYYY-MM-DD) when this model is scheduled for sunset, or null. */
	expiration_date: z.string().nullish(),
});

const catalogResponseSchema = z.object({ data: z.array(openRouterModelSchema) });

const rankingResponseSchema = z.object({
	data: z.object({
		models: z.array(z.object({ slug: z.string() })),
	}),
});

type OpenRouterModel = z.infer<typeof openRouterModelSchema>;

const fetchOpenRouterCatalog = async (): Promise<OpenRouterModel[]> => {
	const res = await fetch(OPENROUTER_URL);
	if (!res.ok) {
		throw new Error(`openrouter /v1/models: HTTP ${res.status} ${await res.text()}`);
	}
	return catalogResponseSchema.parse(await res.json()).data;
};

/**
 * fetches openrouter's `top-weekly` ranking and returns slugs in popularity order. uses the frontend endpoint
 * — undocumented but the only public source of usage-based ranking, mirroring what powers
 * openrouter.ai/rankings.
 */
const fetchOpenRouterRanking = async (): Promise<string[]> => {
	const res = await fetch(OPENROUTER_RANKED_URL);
	if (!res.ok) {
		throw new Error(`openrouter ranking: HTTP ${res.status} ${await res.text()}`);
	}
	return rankingResponseSchema.parse(await res.json()).data.models.map((m) => m.slug);
};

/**
 * true only when the model's sole output is text. catches multimodal generators like gemini-flash-image
 * (`['image', 'text']`) and lyria (`['text', 'audio']`) that emit text as a secondary channel but aren't chat
 * models.
 */
const outputsTextOnly = (m: OpenRouterModel): boolean => {
	const out = m.architecture?.output_modalities ?? [];
	return out.length === 1 && out[0] === 'text';
};

/**
 * drop models scheduled for sunset on or before today. openrouter sets `expiration_date` (ISO `YYYY-MM-DD`)
 * when a provider has announced a deprecation date; the model still answers requests until then.
 */
const isExpired = (m: OpenRouterModel, today: string): boolean => {
	const date = m.expiration_date;
	if (!date) {
		return false;
	}
	return date <= today;
};

const todayISO = (): string => new Date().toISOString().slice(0, 10);

const dedupeAndSort = (ids: string[]): string[] => Array.from(new Set(ids)).sort();

interface CatalogContext {
	catalog: OpenRouterModel[];
	/** slug ordering by `top-weekly` usage; earlier = more popular. */
	rankedSlugs: string[];
}

interface Target {
	name: string;
	file: string;
	typeName: string;
	collect: (ctx: CatalogContext) => string[];
}

const today = todayISO();

/**
 * the `~vendor/...-latest` slugs route to whichever model openrouter currently considers "latest" in a
 * family. their stability isn't guaranteed by the provider, so we exclude them from the autocomplete list —
 * `(string & {})` still lets users pass them explicitly.
 */
const isLive = (m: OpenRouterModel): boolean =>
	outputsTextOnly(m) && !m.id.startsWith('~') && !isExpired(m, today);

const targets: Target[] = [
	{
		name: 'anthropic',
		file: 'anthropic-models.ts',
		typeName: 'AnthropicKnownModelId',
		// `:variant` slugs (`:thinking`, `:free`) are openrouter routing flavors,
		// not native anthropic ids — drop them after the prefix strip.
		collect: ({ catalog }) =>
			dedupeAndSort(
				catalog
					.filter((m) => m.id.startsWith('anthropic/') && !m.id.includes(':') && isLive(m))
					.map((m) => m.id.slice('anthropic/'.length).replace(/\./g, '-')),
			),
	},
	{
		name: 'openai',
		file: 'openai-models.ts',
		typeName: 'OpenAIKnownChatModelId',
		// same rationale as anthropic: `:variant` slugs aren't native openai ids.
		// openrouter brands openai's `chat-latest` alias as `gpt-chat-latest` —
		// strip the prefix back to match the id openai's own api accepts.
		collect: ({ catalog }) =>
			dedupeAndSort(
				catalog
					.filter((m) => m.id.startsWith('openai/') && !m.id.includes(':') && isLive(m))
					.map((m) => m.id.slice('openai/'.length))
					.filter((id) => !/search-preview/.test(id))
					.map((id) => (id === 'gpt-chat-latest' ? 'chat-latest' : id)),
			),
	},
	{
		name: 'openrouter',
		file: 'openrouter-models.ts',
		typeName: 'OpenRouterKnownModelId',
		// emit the top-N most-used models by openrouter's weekly ranking.
		// the catalog has hundreds of entries and most are obscure; ranking
		// keeps autocomplete focused while `(string & {})` covers the rest.
		collect: ({ catalog, rankedSlugs }) => {
			const live = new Set(catalog.filter(isLive).map((m) => m.id));
			const ranked: string[] = [];
			for (const slug of rankedSlugs) {
				if (live.has(slug)) {
					ranked.push(slug);
					if (ranked.length >= OPENROUTER_TOP_N) {
						break;
					}
				}
			}
			return dedupeAndSort(ranked);
		},
	},
];

const renderUnionFile = (typeName: string, ids: string[]): string => {
	const header = [
		'// @generated — do not edit manually.',
		'// run `pnpm update-models` to regenerate.',
		'',
	].join('\n');

	if (ids.length === 0) {
		return `${header}\nexport type ${typeName} = never;\n`;
	}

	const body = [
		`export type ${typeName} =`,
		...ids.map((id, i) => `\t| '${id}'${i === ids.length - 1 ? ';' : ''}`),
	].join('\n');
	return `${header}\n${body}\n`;
};

const main = async () => {
	const onlyArg = process.argv.find((a) => a.startsWith('--only='));
	const only = onlyArg ? new Set(onlyArg.slice('--only='.length).split(',')) : null;

	console.log(`fetching openrouter catalog and ranking...`);
	const [catalog, rankedSlugs] = await Promise.all([fetchOpenRouterCatalog(), fetchOpenRouterRanking()]);
	console.log(`  got ${catalog.length} catalog models, ${rankedSlugs.length} ranked slugs`);

	await mkdir(OUTPUT_DIR, { recursive: true });
	const ctx: CatalogContext = { catalog, rankedSlugs };

	const writes = targets
		.filter((t) => !only || only.has(t.name))
		.map(async (t) => {
			const ids = t.collect(ctx);
			const out = resolve(OUTPUT_DIR, t.file);
			await writeFile(out, renderUnionFile(t.typeName, ids), 'utf-8');
			console.log(`  ${t.name}: wrote ${ids.length} models -> ${out}`);
		});
	await Promise.all(writes);
};

main().catch((err) => {
	console.error(err);
	process.exit(1);
});
