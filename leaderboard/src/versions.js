import { DATA_VERSIONS, DEFAULT_DATA_VERSION } from "./config.js";

export function getAvailableVersions(leaderboardData) {
  const available = new Set();
  for (const version of DATA_VERSIONS) {
    const splits = leaderboardData?.[version] ?? {};
    const hasRows = Object.values(splits).some((rows) => Array.isArray(rows) && rows.length > 0);
    if (hasRows) available.add(version);
  }
  return available;
}

export function versionFromSearch(availableVersions) {
  const requested = new URL(window.location.href).searchParams.get("version");
  if (requested && availableVersions.has(requested)) return requested;
  if (availableVersions.has(DEFAULT_DATA_VERSION)) return DEFAULT_DATA_VERSION;
  return DATA_VERSIONS.find((version) => availableVersions.has(version)) || DEFAULT_DATA_VERSION;
}

function writeVersionToUrl(version, mode = "push") {
  const url = new URL(window.location.href);
  url.searchParams.set("version", version);
  if (mode === "replace") {
    window.history.replaceState({}, "", url);
  } else {
    window.history.pushState({}, "", url);
  }
}

export function initVersionTabs(versionTabs, state, onApply) {
  for (const [version, tab] of versionTabs) {
    if (!tab) continue;
    tab.addEventListener("click", () => {
      if (!state.availableVersions.has(version)) return;
      state.activeVersion = version;
      writeVersionToUrl(version);
      onApply(true);
    });

    tab.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      const available = DATA_VERSIONS.filter((item) => state.availableVersions.has(item));
      const currentIndex = available.indexOf(version);
      if (currentIndex < 0 || available.length < 2) return;
      event.preventDefault();
      const step = event.key === "ArrowRight" ? 1 : -1;
      const nextVersion = available[(currentIndex + step + available.length) % available.length];
      const nextTab = versionTabs.get(nextVersion);
      state.activeVersion = nextVersion;
      writeVersionToUrl(nextVersion);
      nextTab?.focus();
      onApply(true);
    });
  }

  window.addEventListener("popstate", () => {
    state.activeVersion = versionFromSearch(state.availableVersions);
    onApply(true);
  });
}

export function normalizeVersionUrl(state) {
  const requested = new URL(window.location.href).searchParams.get("version");
  if (requested !== state.activeVersion) writeVersionToUrl(state.activeVersion, "replace");
}

export function updateVersionTabsUI(versionTabs, activeVersion, availableVersions) {
  for (const [version, tab] of versionTabs) {
    if (!tab) continue;
    const available = availableVersions.has(version);
    const active = available && version === activeVersion;
    tab.disabled = !available;
    tab.setAttribute("aria-disabled", available ? "false" : "true");
    tab.setAttribute("aria-selected", active ? "true" : "false");
    tab.tabIndex = active ? 0 : -1;
    tab.title = available ? `Show Dataset ${version} results` : `${version.toUpperCase()} results are not available yet`;
  }
}
