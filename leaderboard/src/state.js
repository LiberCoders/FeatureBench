export const state = {
  /** @type {Record<string, Array<any>> | null} */
  leaderboardData: null,

  /** @type {{ allFilters: string[], allTags: string[] } | null} */
  optionsConfig: null,

  /** @type {{ key: 'passed' | 'resolved' | 'date', dir: 'asc' | 'desc' }} */
  sortState: { key: "resolved", dir: "desc" },

  /** @type {Map<string, Set<string>>} */
  selectedTagsBySplit: new Map(),

  /** @type {Set<string> | null} */
  selectedFilters: null,
};
