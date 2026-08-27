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
import { selectEffortRows } from "./efforts.js";
import {
  getAvailableVersions,
  initVersionTabs,
  normalizeVersionUrl,
  updateVersionTabsUI,
  versionFromSearch,
} from "./versions.js";

let footnotesExpanded = false;

function setFootnotesExpanded(expanded) {
  footnotesExpanded = !!expanded;
  if (els.tableFootnotesToggle) {
    els.tableFootnotesToggle.setAttribute("aria-expanded", footnotesExpanded ? "true" : "false");
  }
}

function wireFootnotesToggle() {
  els.tableFootnotesToggle?.addEventListener("click", () => {
    setFootnotesExpanded(!footnotesExpanded);
  });
}

function updateFootnotes(notesUsed) {
  if (!els.tableFootnotes) {
    return;
  }
  const notes = notesUsed instanceof Set ? notesUsed : new Set();
  let hasVisible = false;
  let visibleCount = 0;
  for (const item of els.tableFootnoteItems) {
    const key = item?.dataset?.note;
    const visible = !!key && notes.has(key);
    item.hidden = !visible;
    hasVisible = hasVisible || visible;
    if (visible) {
      visibleCount += 1;
    }
  }
  els.tableFootnotes.hidden = !hasVisible;
  if (!hasVisible) {
    setFootnotesExpanded(false);
    return;
  }
  if (els.tableFootnotesToggle) {
    const label = `Notes (${visibleCount})`;
    els.tableFootnotesToggle.setAttribute("aria-label", label);
    els.tableFootnotesToggle.title = label;
  }
  setFootnotesExpanded(footnotesExpanded);
}

function updateTableSchema(version) {
  if (els.table) els.table.dataset.version = version;
  for (const [headerVersion, header] of els.versionHeaders) {
    if (header) header.hidden = headerVersion !== version;
  }
}

function setActiveSplit(split, updateTagMenu = true) {
  const active = SPLITS.includes(split) ? split : SPLITS[0] || "lite";
  const version = state.activeVersion;
  const rawRows = state.leaderboardData?.[version]?.[active] ?? [];
  const rows =
    version === "v1.1"
      ? selectEffortRows(rawRows, state.selectedEfforts, `${version}:${active}`)
      : rawRows;
  const filtered = getFilteredRows(state, version, active, rows);
  const sorted = getSortedRows(state.sortState, filtered);

  updateVersionTabsUI(els.versionTabs, version, state.availableVersions);
  updateTableSchema(version);
  updateTabsUI(els.tabs, active);
  els.panel.setAttribute("aria-labelledby", `tab-${active}`);
  updateSortUI(els.sortButtons, state.sortState);
  const notesUsed = renderRows(els.tbody, sorted, {
    version,
    onEffortChange(selectionKey, effort) {
      if (!selectionKey) return;
      state.selectedEfforts.set(selectionKey, effort);
      apply(true);
    },
  });
  updateFootnotes(notesUsed);

  if (updateTagMenu && els.tagsBtn?.getAttribute("aria-expanded") === "true") {
    const tags = buildTagsForSplit(state, version, active);
    renderTagMenu(els, state, version, active, tags, els.tagsSearch?.value || "", apply);
  }
}

function apply(updateTagMenu = true) {
  const active = splitFromHash(SPLITS);
  setActiveSplit(active, updateTagMenu);
  return { version: state.activeVersion, split: active };
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
  wireFootnotesToggle();

  try {
    state.optionsConfig = await loadOptions();
    state.leaderboardData = await loadData();
    state.availableVersions = getAvailableVersions(state.leaderboardData);
    state.activeVersion = versionFromSearch(state.availableVersions);
    normalizeVersionUrl(state);

    initDropdowns(els, state, apply);
    initVersionTabs(els.versionTabs, state, apply);
    initTabs(els.tabs, apply);
    initSort(els.sortButtons, state, () => apply(true));

    apply(true);
  } catch (err) {
    // eslint-disable-next-line no-console
    console.error(err);
  }
})();
