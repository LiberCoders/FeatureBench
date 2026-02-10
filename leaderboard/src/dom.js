import { SPLITS } from "./config.js";

export const els = {
  topBadges: /** @type {HTMLElement | null} */ (document.getElementById("topBadges")),
  tbody: /** @type {HTMLElement} */ (document.getElementById("tbody")),
  panel: /** @type {HTMLElement} */ (document.getElementById("panel")),
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
};
