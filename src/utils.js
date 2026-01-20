export function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  const num = Number(value);
  return Number.isInteger(num) ? num.toFixed(0) : num.toFixed(1);
}

export function parseDateValue(value) {
  if (!value) return null;
  const d = new Date(String(value));
  return Number.isNaN(d.getTime()) ? null : d.getTime();
}

export function safeText(value) {
  return value === null || value === undefined || value === "" ? "-" : String(value);
}

export function normalizeFilterArray(value) {
  if (!value) return [];
  if (Array.isArray(value)) return value.map((x) => String(x)).map((s) => s.trim()).filter(Boolean);
  return [String(value)].map((s) => s.trim()).filter(Boolean);
}

export function slugifyKey(label) {
  return String(label || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

export function splitFromHash(splits) {
  const raw = window.location.hash.replace(/^#/, "").trim();
  const v = raw || "lite";
  return splits.includes(v) ? v : "lite";
}
