export function applyTheme(theme, themeToggle) {
  const next = theme === "dark" ? "dark" : "light";
  document.documentElement.dataset.theme = next;
  try {
    localStorage.setItem("featurebench.theme", next);
  } catch {
    // ignore
  }
  const nextLabel = next === "dark" ? "Switch to light theme" : "Switch to dark theme";
  themeToggle.setAttribute("aria-label", nextLabel);
  themeToggle.setAttribute("title", nextLabel);
}

export function initTheme(themeToggle) {
  let stored = null;
  try {
    stored = localStorage.getItem("featurebench.theme");
  } catch {
    stored = null;
  }
  if (stored === "dark" || stored === "light") {
    applyTheme(stored, themeToggle);
    return;
  }
  const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  applyTheme(prefersDark ? "dark" : "light", themeToggle);
}
