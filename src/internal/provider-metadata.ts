import type { ProviderMetadata } from '../types.ts';

/**
 * read a typed bag of provider-specific metadata under the given key. returns undefined when the bag is
 * missing, the key is absent, or the stored value is not an object.
 *
 * @param meta the providerMetadata record to read from
 * @param key the provider name (e.g. `'openai'`, `'anthropic'`)
 */
// oxlint-disable-next-line typescript/no-unnecessary-type-parameters
export const readProviderMeta = <T>(meta: ProviderMetadata | undefined, key: string): T | undefined => {
	const value = meta?.[key];
	if (!value || typeof value !== 'object') {
		return undefined;
	}
	// oxlint-disable-next-line typescript/no-unsafe-type-assertion
	return value as T;
};
