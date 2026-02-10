import { SPLITS } from "./config.js";
import { els } from "./dom.js";
import { state } from "./state.js";
import { loadData, loadOptions } from "./data.js";
import { initTheme, applyTheme } from "./theme.js";
import { getSortedRows, updateSortUI, initSort } from "./sort.js";
import { getFilteredRows, initDropdowns, buildTagsForSplit, renderTagMenu } from "./filters.js";
import { splitFromHash } from "./utils.js";
import { initTabs, updateTabsUI } from "./tabs.js";
import { renderRows } from "./table.js";

function setActiveSplit(split, updateTagMenu = true) {
  const active = SPLITS.includes(split) ? split : "lite";
  const rows = state.leaderboardData?.[active] ?? [];
  const filtered = getFilteredRows(state, active, rows);
  const sorted = getSortedRows(state.sortState, filtered);

  updateTabsUI(els.tabs, active);
  els.panel.setAttribute("aria-labelledby", `tab-${active}`);
  updateSortUI(els.sortButtons, state.sortState);
  renderRows(els.tbody, sorted);

  if (updateTagMenu && els.tagsBtn?.getAttribute("aria-expanded") === "true") {
    const tags = buildTagsForSplit(state, active);
    renderTagMenu(els, state, active, tags, els.tagsSearch?.value || "", apply);
  }
}

function apply(updateTagMenu = true) {
  const active = splitFromHash(SPLITS);
  setActiveSplit(active, updateTagMenu);
  return active;
}

function wireThemeButtons() {
  const controls = { lightBtn: els.themeLight, darkBtn: els.themeDark };
  els.themeLight?.addEventListener("click", () => {
    applyTheme("light", controls);
  });
  els.themeDark?.addEventListener("click", () => {
    applyTheme("dark", controls);
  });
}

(async () => {
  initTheme({ lightBtn: els.themeLight, darkBtn: els.themeDark });
  wireThemeButtons();

  try {
    state.optionsConfig = await loadOptions();
    state.leaderboardData = await loadData();

    initDropdowns(els, state, apply);
    initTabs(els.tabs, apply);
    initSort(els.sortButtons, state, () => apply(true));

    apply(true);
  } catch (err) {
    // eslint-disable-next-line no-console
    console.error(err);
  }
})();
