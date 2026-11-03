/**
 * RDNA-FAST: the Status page's entry in the chat sidebar's menu (New chat, Search, Models, MCP,
 * Status, Settings). The sidebar opens it like any route item and marks it active on #/status.
 */

import { RDNA_ROUTE_ID_STATUS, RDNA_ROUTE_STATUS } from './constants';
import { Gauge } from '@lucide/svelte';
import type { DesktopIconStripItem } from '$lib/types';

export const RDNA_SIDEBAR_STATUS_ITEM: DesktopIconStripItem = {
	activeRouteId: RDNA_ROUTE_ID_STATUS,
	icon: Gauge,
	route: RDNA_ROUTE_STATUS,
	tooltip: 'Status'
};
