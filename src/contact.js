function applyThemeWithoutToggle(theme) {
  const next = theme === "dark" ? "dark" : "light";
  document.documentElement.dataset.theme = next;
}

function initThemeFromStorageOrSystem() {
  /** @type {string | null} */
  let stored = null;
  try {
    stored = localStorage.getItem("featurebench.theme");
  } catch {
    stored = null;
  }
  if (stored === "dark" || stored === "light") {
    applyThemeWithoutToggle(stored);
    return;
  }

  const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  applyThemeWithoutToggle(prefersDark ? "dark" : "light");
}

initThemeFromStorageOrSystem();
