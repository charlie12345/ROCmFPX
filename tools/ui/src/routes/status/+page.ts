import type { PageLoad } from './$types';
import { initStores } from '$lib/stores/init';

// the status page asks for the API key itself on a 401, so it only waits for the stored settings
export const load: PageLoad = async () => {
	await initStores();
};
