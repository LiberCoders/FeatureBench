import { DATA_VERSIONS, SPLITS } from "./config.js";

export const els = {
  topBadges: /** @type {HTMLElement | null} */ (document.getElementById("topBadges")),
  tbody: /** @type {HTMLElement} */ (document.getElementById("tbody")),
  table: /** @type {HTMLTableElement | null} */ (document.getElementById("leaderboardTable")),
  panel: /** @type {HTMLElement} */ (document.getElementById("panel")),
  tableFootnotes: /** @type {HTMLElement | null} */ (document.getElementById("tableFootnotes")),
  tableFootnotesToggle: /** @type {HTMLButtonElement | null} */ (document.getElementById("tableFootnotesToggle")),
  tableFootnotesContent: /** @type {HTMLElement | null} */ (document.getElementById("tableFootnotesContent")),
  tableFootnoteItems: Array.from(document.querySelectorAll(".table-footnote-item")),
  themeLight: /** @type {HTMLButtonElement | null} */ (document.getElementById("themeLight")),
  themeDark: /** @type {HTMLButtonElement | null} */ (document.getElementById("themeDark")),
  sortButtons: Array.from(document.querySelectorAll(".sort-btn")),

  filtersBtn: /** @type {HTMLButtonElement | null} */ (document.getElementById("filtersBtn")),
  filtersMenu: document.getElementById("filtersMenu"),
  filtersList: document.getElementById("filtersList"),

  tagsBtn: /** @type {HTMLButtonElement | null} */ (document.getElementById("tagsBtn")),
  tagsMenu: document.getElementById("tagsMenu"),
  tagsSearch: /** @type {HTMLInputElement | null} */ (document.getElementById("tagsSearch")),
  tagsList: document.getElementById("tagsList"),
  tagsEmpty: document.getElementById("tagsEmpty"),

  tabs: new Map(SPLITS.map((s) => [s, document.querySelector(`.tab[data-split="${s}"]`)])),
  versionTabs: new Map(
    DATA_VERSIONS.map((version) => [version, document.querySelector(`.dataset-version-tab[data-version="${version}"]`)]),
  ),
  versionHeaders: new Map(
    DATA_VERSIONS.map((version) => [version, document.querySelector(`[data-version-head="${version}"]`)]),
  ),
};
