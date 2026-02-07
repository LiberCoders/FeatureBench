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

const CITATIONS = {
  bibtex: `ABCABCABC`,
  apa: "ABCABCABC",
  mla: "ABCABCABC",
};

/** @type {HTMLElement | null} */
const citeText = document.getElementById("citeText");
/** @type {HTMLButtonElement | null} */
const copyBtn = document.getElementById("copyBtn");
const tabs = Array.from(document.querySelectorAll(".cite-tab"));

function setFormat(nextFormat) {
  const format = nextFormat in CITATIONS ? nextFormat : "bibtex";
  for (const btn of tabs) {
    btn.setAttribute("aria-selected", btn.dataset.format === format ? "true" : "false");
  }
  if (citeText) citeText.textContent = CITATIONS[format];
  if (copyBtn) copyBtn.dataset.format = format;
}

async function copyToClipboard(text) {
  if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
    await navigator.clipboard.writeText(text);
    return;
  }
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.setAttribute("readonly", "");
  ta.style.position = "fixed";
  ta.style.left = "-9999px";
  document.body.appendChild(ta);
  ta.select();
  document.execCommand("copy");
  ta.remove();
}

for (const btn of tabs) {
  btn.addEventListener("click", () => setFormat(btn.dataset.format || "bibtex"));
}

if (copyBtn) {
  copyBtn.addEventListener("click", async () => {
    const format = copyBtn.dataset.format || "bibtex";
    const text = CITATIONS[format] || "";
    try {
      await copyToClipboard(text);
    } catch {
      // ignore
    }
  });
}

setFormat("bibtex");
