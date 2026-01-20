import { normalizeFilterArray, slugifyKey } from "./utils.js";

function setMenuOpen(btn, menu, open) {
  if (!btn || !menu) return;
  btn.setAttribute("aria-expanded", open ? "true" : "false");
  menu.hidden = !open;
}

function closeAllMenus(els) {
  setMenuOpen(els.filtersBtn, els.filtersMenu, false);
  setMenuOpen(els.tagsBtn, els.tagsMenu, false);
}

function ensureAllFiltersSelected(state, allFilterLabels) {
  if (state.selectedFilters === null) state.selectedFilters = new Set(allFilterLabels);
}

function computeAllFiltersState(state, allFilterLabels) {
  const total = allFilterLabels.length;
  let count = 0;
  const selected = state.selectedFilters ?? new Set();
  for (const f of allFilterLabels) if (selected.has(f)) count++;
  return { total, count, all: count === total, none: count === 0 };
}

export function renderFiltersMenu(els, state, onApply) {
  if (!els.filtersList || !state.optionsConfig) return;
  const allFilterLabels = Array.isArray(state.optionsConfig.allFilters) ? state.optionsConfig.allFilters.slice() : [];

  ensureAllFiltersSelected(state, allFilterLabels);
  const selected = state.selectedFilters ?? new Set();
  els.filtersList.textContent = "";

  const mkRow = (labelText, dataKey, checked, onChange) => {
    const label = document.createElement("label");
    label.className = "dd-item";
    const input = document.createElement("input");
    input.type = "checkbox";
    if (dataKey) input.dataset.filter = dataKey;
    input.checked = !!checked;
    input.addEventListener("change", onChange);
    const span = document.createElement("span");
    span.textContent = labelText;
    label.append(input, span);
    return { label, input };
  };

  const st0 = computeAllFiltersState(state, allFilterLabels);
  const all = mkRow("All Filters", "all", st0.all, () => {
    state.selectedFilters = all.input.checked ? new Set(allFilterLabels) : new Set();
    renderFiltersMenu(els, state, onApply);
    onApply(true);
  });
  els.filtersList.appendChild(all.label);
  all.input.indeterminate = !st0.all && !st0.none;

  for (const f of allFilterLabels) {
    const key = slugifyKey(f);
    const row = mkRow(f, key, selected.has(f), () => {
      if (state.selectedFilters === null) state.selectedFilters = new Set();
      if (row.input.checked) state.selectedFilters.add(f);
      else state.selectedFilters.delete(f);
      const st = computeAllFiltersState(state, allFilterLabels);
      all.input.checked = st.all;
      all.input.indeterminate = !st.all && !st.none;
      onApply(true);
    });
    els.filtersList.appendChild(row.label);
  }

  const st = computeAllFiltersState(state, allFilterLabels);
  all.input.checked = st.all;
  all.input.indeterminate = !st.all && !st.none;
}

function ensureAllTagsSelected(state, split, allItems) {
  if (!state.selectedTagsBySplit.has(split)) {
    state.selectedTagsBySplit.set(split, new Set(allItems));
  }
}

function computeAllTagsState(selected, allItems) {
  const total = allItems.length;
  let count = 0;
  for (const t of allItems) if (selected.has(t)) count++;
  return { total, count, all: count === total, none: count === 0 };
}

export function buildTagsForSplit(state, split) {
  const rows = state.leaderboardData?.[split] ?? [];
  const out = [];
  for (const r of rows) {
    const vals = normalizeFilterArray(r?.filter_2);
    for (const v of vals) out.push(v);
  }
  return Array.from(new Set(out)).sort((a, b) => a.localeCompare(b));
}

export function renderTagMenu(els, state, split, allItems, query, onApply) {
  if (!els.tagsList || !els.tagsEmpty) return;

  ensureAllTagsSelected(state, split, allItems);
  const selected = state.selectedTagsBySplit.get(split);

  const q = (query || "").trim().toLowerCase();
  const filtered = q ? allItems.filter((t) => t.toLowerCase().includes(q)) : allItems;

  els.tagsList.textContent = "";

  const allRow = document.createElement("label");
  allRow.className = "dd-item";
  const allInput = document.createElement("input");
  allInput.type = "checkbox";
  allInput.dataset.tag = "all";
  const allSpan = document.createElement("span");
  allSpan.textContent = "All Tags";
  allRow.append(allInput, allSpan);
  els.tagsList.appendChild(allRow);

  const syncAllCheckbox = () => {
    const st = computeAllTagsState(selected, allItems);
    allInput.checked = st.all;
    allInput.indeterminate = !st.all && !st.none;
  };

  allInput.addEventListener("change", () => {
    if (allInput.checked) {
      state.selectedTagsBySplit.set(split, new Set(allItems));
    } else {
      state.selectedTagsBySplit.set(split, new Set());
    }
    const nextSelected = state.selectedTagsBySplit.get(split);
    selected.clear();
    for (const t of nextSelected) selected.add(t);
    renderTagMenu(els, state, split, allItems, els.tagsSearch?.value || "", onApply);
    onApply(false);
  });

  for (const t of filtered) {
    const label = document.createElement("label");
    label.className = "dd-item";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = selected.has(t);
    input.addEventListener("change", () => {
      if (input.checked) selected.add(t);
      else selected.delete(t);
      syncAllCheckbox();
      onApply(false);
    });
    const span = document.createElement("span");
    span.textContent = t;
    label.append(input, span);
    els.tagsList.appendChild(label);
  }

  syncAllCheckbox();

  const empty = filtered.length === 0;
  els.tagsEmpty.hidden = !empty;
}

export function rowMatchesSelectedFilters(state, row) {
  if (!state.optionsConfig) return true;
  const allFilterLabels = Array.isArray(state.optionsConfig.allFilters) ? state.optionsConfig.allFilters : [];
  if (allFilterLabels.length === 0) return true;
  ensureAllFiltersSelected(state, allFilterLabels);
  if (!state.selectedFilters || state.selectedFilters.size === 0) return false;
  const rowFilters = normalizeFilterArray(row?.filter_1);
  for (const f of state.selectedFilters) {
    if (rowFilters.includes(f)) return true;
  }
  return false;
}

export function rowMatchesSelectedTags(state, split, row) {
  const allItems = buildTagsForSplit(state, split);
  ensureAllTagsSelected(state, split, allItems);
  const selected = state.selectedTagsBySplit.get(split);
  if (!selected || selected.size === 0) return false;
  const rowTags = normalizeFilterArray(row?.filter_2);
  for (const t of selected) {
    if (rowTags.includes(t)) return true;
  }
  return false;
}

export function getFilteredRows(state, split, rows) {
  const input = Array.isArray(rows) ? rows : [];
  return input.filter((r) => rowMatchesSelectedFilters(state, r) && rowMatchesSelectedTags(state, split, r));
}

export function initDropdowns(els, state, onApply) {
  const onDocClick = (e) => {
    const target = /** @type {Node} */ (e.target);
    const withinFilters = els.filtersBtn?.contains(target) || els.filtersMenu?.contains(target);
    const withinTags = els.tagsBtn?.contains(target) || els.tagsMenu?.contains(target);
    if (withinFilters || withinTags) return;
    closeAllMenus(els);
  };

  document.addEventListener("click", onDocClick);
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    closeAllMenus(els);
  });

  els.filtersBtn?.addEventListener("click", () => {
    const open = els.filtersBtn.getAttribute("aria-expanded") !== "true";
    closeAllMenus(els);
    setMenuOpen(els.filtersBtn, els.filtersMenu, open);
  });

  renderFiltersMenu(els, state, onApply);

  els.tagsBtn?.addEventListener("click", () => {
    const open = els.tagsBtn.getAttribute("aria-expanded") !== "true";
    closeAllMenus(els);
    setMenuOpen(els.tagsBtn, els.tagsMenu, open);

    if (open) {
      const split = onApply(true);
      const tags = buildTagsForSplit(state, split);
      renderTagMenu(els, state, split, tags, els.tagsSearch?.value || "", onApply);
      els.tagsSearch?.focus();
    }
  });

  els.tagsSearch?.addEventListener("input", () => {
    const split = onApply(true);
    const tags = buildTagsForSplit(state, split);
    renderTagMenu(els, state, split, tags, els.tagsSearch.value, onApply);
  });
}
