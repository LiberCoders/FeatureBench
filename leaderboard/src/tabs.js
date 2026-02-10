import { SPLITS } from "./config.js";

export function initTabs(tabs, onApply) {
  for (const [split, tab] of tabs) {
    if (!tab) continue;
    tab.addEventListener("click", () => {
      window.location.hash = split;
    });
    tab.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
      e.preventDefault();
      const idx = SPLITS.indexOf(split);
      const nextIdx = e.key === "ArrowRight" ? (idx + 1) % SPLITS.length : (idx - 1 + SPLITS.length) % SPLITS.length;
      const nextSplit = SPLITS[nextIdx];
      const nextTab = tabs.get(nextSplit);
      nextTab?.focus();
      window.location.hash = nextSplit;
    });
  }

  window.addEventListener("hashchange", () => {
    onApply(true);
  });
}

export function updateTabsUI(tabs, active) {
  for (const s of SPLITS) {
    const tab = tabs.get(s);
    if (!tab) continue;
    tab.setAttribute("aria-selected", s === active ? "true" : "false");
    tab.tabIndex = s === active ? 0 : -1;
  }
}
