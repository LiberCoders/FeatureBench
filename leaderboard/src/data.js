import { DATA_URLS, OPTIONS_URL, SPLITS, TOP_BADGES_URL } from "./config.js";

export async function loadData() {
  const pairs = await Promise.all(
    SPLITS.map(async (split) => {
      const url = DATA_URLS[split];
      const res = await fetch(url, { cache: "no-store" });
      if (!res.ok) throw new Error(`Failed to load ${url}: ${res.status}`);
      const json = await res.json();
      if (!Array.isArray(json)) throw new Error(`Invalid JSON for ${url}: expected an array`);
      return [split, json];
    }),
  );
  return Object.fromEntries(pairs);
}

export async function loadOptions() {
  const res = await fetch(OPTIONS_URL, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to load ${OPTIONS_URL}: ${res.status}`);
  const json = await res.json();
  const allFilters = Array.isArray(json?.allFilters) ? json.allFilters.map((s) => String(s)) : [];
  const allTags = Array.isArray(json?.allTags) ? json.allTags.map((s) => String(s)) : [];
  return { allFilters, allTags };
}

export async function loadTopBadges() {
  const res = await fetch(TOP_BADGES_URL, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to load ${TOP_BADGES_URL}: ${res.status}`);
  const json = await res.json();
  if (!json || typeof json !== "object") throw new Error(`Invalid JSON for ${TOP_BADGES_URL}: expected object`);
  return json;
}
