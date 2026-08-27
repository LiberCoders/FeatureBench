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

function textList(value) {
  const values = Array.isArray(value) ? value : value ? [value] : [];
  const normalized = values.map((item) => safeText(item).trim()).filter((item) => item && item !== "-");
  return normalized.length ? normalized.join(" · ") : "-";
}

function createEffortCell(row, rowLabel, onEffortChange) {
  const td = document.createElement("td");
  td.className = "col-effort";
  const options = Array.isArray(row.__effortOptions) ? row.__effortOptions : [];
  const current = safeText(row.effort);

  if (options.length <= 1) {
    td.textContent = current;
    return td;
  }

  const select = document.createElement("select");
  select.className = "effort-select";
  select.setAttribute("aria-label", `Effort for ${rowLabel}`);
  for (const effort of options) {
    const option = document.createElement("option");
    option.value = effort;
    option.textContent = effort;
    option.selected = effort === current;
    select.appendChild(option);
  }
  select.addEventListener("change", () => {
    onEffortChange?.(row.__effortSelectionKey, select.value);
  });
  td.appendChild(select);
  return td;
}

function renderEmptyRow(tbody, version) {
  const tr = document.createElement("tr");
  const td = document.createElement("td");
  td.className = "empty-results";
  td.colSpan = version === "v1.1" ? 9 : 8;
  td.textContent = "No results available for this split.";
  tr.appendChild(td);
  tbody.appendChild(tr);
}

export function renderRows(tbody, rows, options = {}) {
  tbody.textContent = "";
  const notesUsed = new Set();
  const version = options.version === "v1.1" ? "v1.1" : "v1.0";

  if (!rows.length) {
    renderEmptyRow(tbody, version);
    return notesUsed;
  }

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

    const tdDate = document.createElement("td");
    tdDate.className = "col-date";
    tdDate.textContent = safeText(row.date);

    if (version === "v1.1") {
      const tdEffort = createEffortCell(row, rowLabel, options.onEffortChange);
      const tdAgentOrg = document.createElement("td");
      tdAgentOrg.className = "col-agent-org";
      tdAgentOrg.textContent = textList(row.agent_org ?? row.agentOrg);

      const tdModelOrg = document.createElement("td");
      tdModelOrg.className = "col-model-org";
      tdModelOrg.textContent = textList(row.model_org ?? row.modelOrg);

      tr.append(tdRank, tdModel, tdAgent, tdEffort, tdPassed, tdResolved, tdAgentOrg, tdModelOrg, tdDate);
    } else {
      const tdAgentOrg = document.createElement("td");
      tdAgentOrg.className = "col-agent-org";
      tdAgentOrg.textContent = textList(row.agent_org ?? row.agentOrg);

      const tdModelOrg = document.createElement("td");
      tdModelOrg.className = "col-model-org";
      tdModelOrg.textContent = textList(row.model_org ?? row.modelOrg);

      tr.append(tdRank, tdModel, tdAgent, tdPassed, tdResolved, tdAgentOrg, tdModelOrg, tdDate);
    }
    tbody.appendChild(tr);
  }

  return notesUsed;
}
