export const state = {
  /** @type {Record<string, Record<string, Array<any>>> | null} */
  leaderboardData: null,

  /** @type {{ allFilters: string[], allTags: string[] } | null} */
  optionsConfig: null,

  /** @type {{ key: 'passed' | 'resolved' | 'model_release', dir: 'asc' | 'desc' }} */
  sortState: { key: "passed", dir: "desc" },

  /** @type {Map<string, Set<string>>} */
  selectedTagsBySplit: new Map(),

  /** @type {Map<string, string>} */
  selectedEfforts: new Map(),

  /** @type {Set<string>} */
  availableVersions: new Set(),

  /** @type {string} */
  activeVersion: "v1.0",

  /** @type {Set<string> | null} */
  selectedFilters: null,
};
