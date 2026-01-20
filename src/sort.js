import { parseDateValue } from "./utils.js";

export function compareBy(key, dir) {
  const sign = dir === "asc" ? 1 : -1;
  return (a, b) => {
    if (key === "date") {
      const av = parseDateValue(a.date);
      const bv = parseDateValue(b.date);
      if (av === null && bv === null) return 0;
      if (av === null) return 1;
      if (bv === null) return -1;
      return (av - bv) * sign;
    }

    const av = Number(a[key]);
    const bv = Number(b[key]);
    const aBad = a[key] === null || a[key] === undefined || Number.isNaN(av);
    const bBad = b[key] === null || b[key] === undefined || Number.isNaN(bv);
    if (aBad && bBad) return 0;
    if (aBad) return 1;
    if (bBad) return -1;
    return (av - bv) * sign;
  };
}

export function getSortedRows(sortState, rows) {
  const copied = Array.isArray(rows) ? rows.slice() : [];
  copied.sort(compareBy(sortState.key, sortState.dir));
  return copied;
}

export function updateSortUI(sortButtons, sortState) {
  for (const btn of sortButtons) {
    const key = btn.dataset.sort;
    const active = key === sortState.key;
    btn.setAttribute("aria-pressed", active ? "true" : "false");
    btn.dataset.dir = active ? sortState.dir : "desc";
  }
}

export function initSort(sortButtons, state, onApply) {
  for (const btn of sortButtons) {
    btn.addEventListener("click", () => {
      const key = btn.dataset.sort;
      if (key !== "passed" && key !== "resolved" && key !== "date") return;
      if (state.sortState.key === key) {
        state.sortState.dir = state.sortState.dir === "desc" ? "asc" : "desc";
      } else {
        state.sortState = { key, dir: "desc" };
      }
      onApply();
    });
  }
}
