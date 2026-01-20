function safeStr(v) {
  return v === null || v === undefined ? "" : String(v);
}

function normalizeTone(tone) {
  const t = safeStr(tone).trim().toLowerCase();
  if (
    t === "success" ||
    t === "info" ||
    t === "danger" ||
    t === "neutral" ||
    t === "arxiv" ||
    t === "github" ||
    t === "huggingface"
  )
    return t;
  return "neutral";
}

function buildBadge(item) {
  const label = safeStr(item?.label).trim();
  const value = safeStr(item?.value).trim();
  const href = safeStr(item?.href).trim();
  if (!label || !value || !href) return null;

  const a = document.createElement("a");
  a.className = `badge badge-${normalizeTone(item?.tone)}`;
  a.href = href;
  a.target = "_blank";
  a.rel = "noreferrer";
  a.title = href;
  a.setAttribute("aria-label", `Open ${label}: ${value}`);

  const s1 = document.createElement("span");
  s1.className = "badge-label";
  s1.textContent = label;

  const s2 = document.createElement("span");
  s2.className = "badge-value";
  s2.textContent = value;

  a.append(s1, s2);
  return a;
}

export function renderTopBadges(container, badgeConfig) {
  if (!container) return;
  container.textContent = "";

  const items = Array.isArray(badgeConfig?.items) ? badgeConfig.items : [];
  for (const item of items) {
    const el = buildBadge(item);
    if (el) container.appendChild(el);
  }
}
