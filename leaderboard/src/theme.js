function syncButtons(theme, controls) {
  const lightBtn = controls?.lightBtn;
  const darkBtn = controls?.darkBtn;
  if (!lightBtn || !darkBtn) {
    return;
  }
  lightBtn.classList.toggle("active", theme === "light");
  darkBtn.classList.toggle("active", theme === "dark");
  lightBtn.setAttribute("aria-pressed", theme === "light" ? "true" : "false");
  darkBtn.setAttribute("aria-pressed", theme === "dark" ? "true" : "false");
}

export function applyTheme(theme, controls = null) {
  const next = theme === "dark" ? "dark" : "light";
  document.documentElement.dataset.theme = next;
  syncButtons(next, controls);
  try {
    localStorage.setItem("featurebench.theme", next);
  } catch {
    // ignore
  }
}

export function initTheme(controls = null) {
  let stored = null;
  try {
    stored = localStorage.getItem("featurebench.theme");
  } catch {
    stored = null;
  }
  if (stored === "dark" || stored === "light") {
    applyTheme(stored, controls);
    return;
  }
  const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  applyTheme(prefersDark ? "dark" : "light", controls);
}
