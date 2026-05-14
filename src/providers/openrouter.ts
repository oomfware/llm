import type { ChatAdapter } from '../adapter.ts';

import type { OpenRouterKnownModelId } from './generated/openrouter-models.ts';
import { openaiCompatible, type OpenAICompatibleProviderOptions } from './openai-compatible.ts';

/**
 * known OpenRouter model slugs. the `(string & {})` tail keeps autocomplete working for known entries while
 * still accepting any other model string — OpenRouter's catalog grows constantly. the literal union is
 * generated — see `scripts/update-models.ts`.
 */
export type OpenRouterModel = OpenRouterKnownModelId | (string & {});

/**
 * OpenRouter routing/provider preferences. mirrors the upstream JSON shape — see
 * https://openrouter.ai/docs/features/provider-routing.
 */
export interface OpenRouterProviderPreferences {
	order?: string[];
	allow_fallbacks?: boolean;
	require_parameters?: boolean;
	data_collection?: 'allow' | 'deny';
	ignore?: string[];
	quantizations?: string[];
	sort?: 'price' | 'throughput' | 'latency';
}

export interface OpenRouterProviderOptions extends OpenAICompatibleProviderOptions {
	/** fallback model slugs to try if the primary model fails. */
	models?: string[];
	/** routing strategy. `'fallback'` enables `models` fallback behavior. */
	route?: 'fallback';
	/** provider-routing preferences. */
	provider?: OpenRouterProviderPreferences;
	/** input transforms, e.g. `['middle-out']` for long-context compression. */
	transforms?: string[];
}

export interface OpenRouterConfig {
	apiKey?: string;
	baseUrl?: string;
	headers?: Record<string, string>;
	fetch?: typeof fetch;
	/** site/app name; sent as `X-Title` to appear on the OpenRouter dashboard. */
	appName?: string;
	/** site url; sent as `HTTP-Referer` for attribution and ranking. */
	referrer?: string;
}

export type OpenRouterAdapter<TModel extends OpenRouterModel = OpenRouterModel> = ChatAdapter<
	TModel,
	OpenRouterProviderOptions
>;

/**
 * create an OpenRouter chat adapter. talks to the OpenAI-compatible Chat Completions endpoint at
 * `openrouter.ai/api/v1`, with extra provider-routing fields (`models`, `route`, `provider`, `transforms`)
 * available via `providerOptions`.
 */
export const openrouter = <const TModel extends OpenRouterModel>(
	model: TModel,
	config: OpenRouterConfig = {},
): OpenRouterAdapter<TModel> => {
	const headers: Record<string, string> = { ...config.headers };
	if (config.appName) {
		headers['x-title'] = config.appName;
	}
	if (config.referrer) {
		headers['http-referer'] = config.referrer;
	}

	// oxlint-disable-next-line typescript/no-unnecessary-type-arguments
	return openaiCompatible<TModel, OpenRouterProviderOptions>(model, {
		name: 'openrouter',
		baseUrl: config.baseUrl ?? 'https://openrouter.ai/api/v1',
		apiKey: config.apiKey,
		headers,
		fetch: config.fetch,
		extendBody: (opts) => {
			const out: Record<string, unknown> = {};
			if (opts.models !== undefined) {
				out.models = opts.models;
			}
			if (opts.route !== undefined) {
				out.route = opts.route;
			}
			if (opts.provider !== undefined) {
				out.provider = opts.provider;
			}
			if (opts.transforms !== undefined) {
				out.transforms = opts.transforms;
			}
			return out;
		},
	});
};
