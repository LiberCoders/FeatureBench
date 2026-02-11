import { formatPercent, safeText } from "./utils.js";

const AGENT_MARKERS = [
  {
    key: "routing",
    token: "(routing)",
    sup: "1",
    note: "Routing: Routing mode may route operations to different models, even if a specific model is selected.",
  },
  {
    key: "mock",
    token: "(mock)",
    sup: "2",
    note: "Mock: Tool calls are string-encoded and simulated.",
  },
];

function getAgentAndModel(row) {
  const agent = safeText(row.agent).trim();
  const model = safeText(row.model).trim();
  if (agent) return { agent, model };

  // Backward compatibility for older data where `model` is "agent + model".
  const m = model.match(/^(.*?)\s\+\s(.+)$/);
  if (m) {
    return { agent: safeText(m[1]).trim(), model: safeText(m[2]).trim() };
  }
  return { agent: "-", model };
}

function parseAgentLabel(rawAgent) {
  let text = safeText(rawAgent).trim();
  const used = [];
  const lower = text.toLowerCase();

  for (const marker of AGENT_MARKERS) {
    if (lower.includes(marker.token)) {
      used.push(marker);
      const pattern = new RegExp(`\\s*\\(${marker.key}\\)\\s*`, "gi");
      text = text.replace(pattern, " ");
    }
  }

  text = text.replace(/\s+/g, " ").trim();
  if (!text) {
    text = "-";
  }
  return { text, used };
}

export function renderRows(tbody, rows) {
  tbody.textContent = "";
  const notesUsed = new Set();

  for (const [index, row] of rows.entries()) {
    const tr = document.createElement("tr");

    const { agent, model } = getAgentAndModel(row);
    const agentLabel = parseAgentLabel(agent);
    const rowLabel = agentLabel.text && model ? `${agentLabel.text} + ${model}` : model || agentLabel.text || "-";

    const tdRank = document.createElement("td");
    tdRank.className = "col-rank";
    tdRank.textContent = String(index + 1);

    const tdAgent = document.createElement("td");
    tdAgent.className = "col-agent";
    tdAgent.textContent = agentLabel.text;
    for (const marker of agentLabel.used) {
      notesUsed.add(marker.key);
      const sup = document.createElement("sup");
      sup.className = "agent-note-sup";
      sup.textContent = marker.sup;
      if (marker.note) {
        // Styled tooltip content rendered via CSS pseudo element.
        sup.dataset.tooltip = marker.note;
        sup.setAttribute("aria-label", marker.note);
        sup.tabIndex = 0;
      }
      tdAgent.appendChild(sup);
    }

    const tdModel = document.createElement("td");
    tdModel.className = "col-model";
    tdModel.textContent = safeText(model);

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
      a.setAttribute("aria-label", `Open link for ${rowLabel}`);
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

    tr.append(tdRank, tdModel, tdAgent, tdResolved, tdPassed, tdOrg, tdDate, tdSite);
    tbody.appendChild(tr);
  }

  return notesUsed;
}
