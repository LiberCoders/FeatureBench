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
  bibtex: `@article{zhou2026featurebench,
  title={FeatureBench: Benchmarking Agentic Coding for Complex Feature Development},
  author={Zhou, Qixing and Zhang, Jiacheng and Wang, Haiyang and Hao, Rui and Wang, Jiahe and Han, Minghao and Yang, Yuxue and Wu, Shuzhe and Pan, Feiyang and Fan, Lue and others},
  journal={arXiv preprint arXiv:2602.10975},
  year={2026}
}`,
  apa: "Zhou, Q., Zhang, J., Wang, H., Hao, R., Wang, J., Han, M., Yang, Y., Wu, S., Pan, F., Fan, L., Tu, D., & Zhang, Z. (2026). FeatureBench: Benchmarking agentic coding for complex feature development. arXiv. https://doi.org/10.48550/arXiv.2602.10975",
  mla: "Zhou, Qixing, et al. “FeatureBench: Benchmarking Agentic Coding for Complex Feature Development.” arXiv, 2026, https://doi.org/10.48550/arXiv.2602.10975.",
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
