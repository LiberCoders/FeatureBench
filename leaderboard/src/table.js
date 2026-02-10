import { formatPercent, safeText } from "./utils.js";

export function renderRows(tbody, rows) {
  tbody.textContent = "";

  for (const row of rows) {
    const tr = document.createElement("tr");

    const tdModel = document.createElement("td");
    tdModel.className = "col-model";
    tdModel.textContent = safeText(row.model);

    const tdPassed = document.createElement("td");
    tdPassed.className = "num pct score";
    tdPassed.textContent = formatPercent(row.passed);

    const tdResolved = document.createElement("td");
    tdResolved.className = "num pct score";
    tdResolved.textContent = formatPercent(row.resolved);

    const tdOrg = document.createElement("td");
    tdOrg.className = "col-org";

    const orgValue = row.org;
    const orgPaths = Array.isArray(orgValue) ? orgValue : orgValue ? [orgValue] : [];

    if (orgPaths.length === 0) {
      tdOrg.textContent = "-";
    } else {
      const wrap = document.createElement("div");
      wrap.className = "org-logos";
      for (const p of orgPaths) {
        const src = p ? String(p) : "";
        if (!src) continue;
        const img = document.createElement("img");
        img.className = "org-logo";
        img.loading = "lazy";
        img.decoding = "async";
        img.src = src;
        img.alt = "Org logo";
        img.addEventListener("error", () => {
          img.remove();
        });
        wrap.appendChild(img);
      }
      if (wrap.childElementCount === 0) {
        tdOrg.textContent = "-";
      } else {
        tdOrg.appendChild(wrap);
      }
    }

    const tdDate = document.createElement("td");
    tdDate.className = "col-date";
    tdDate.textContent = safeText(row.date);

    const tdSite = document.createElement("td");
    tdSite.className = "site col-site";

    const siteUrl = row.siteUrl ? String(row.siteUrl) : "";
    if (siteUrl) {
      const a = document.createElement("a");
      a.href = siteUrl;
      a.target = "_blank";
      a.rel = "noreferrer";
      a.setAttribute("aria-label", `Open link for ${safeText(row.model)}`);
      a.title = "Open";
      a.innerHTML =
        '<svg class="site-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">' +
        '<path d="M14 3h7v7" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />' +
        '<path d="M21 3l-9 9" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />' +
        '<path d="M10 7H7a4 4 0 0 0-4 4v6a4 4 0 0 0 4 4h6a4 4 0 0 0 4-4v-3" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />' +
        "</svg>";
      tdSite.appendChild(a);
    } else {
      tdSite.textContent = "-";
    }

    tr.append(tdModel, tdResolved, tdPassed, tdOrg, tdDate, tdSite);
    tbody.appendChild(tr);
  }
}
