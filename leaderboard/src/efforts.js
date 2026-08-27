function normalizedText(value) {
  return value === null || value === undefined ? "" : String(value).trim();
}

function groupKey(row) {
  return JSON.stringify([normalizedText(row?.agent), normalizedText(row?.model)]);
}

function effortLabel(row) {
  return normalizedText(row?.effort) || "-";
}

export function selectEffortRows(rows, selectedEfforts, scope) {
  const groups = new Map();
  for (const row of Array.isArray(rows) ? rows : []) {
    const key = groupKey(row);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }

  const selectedRows = [];
  for (const [key, variants] of groups) {
    const optionsByLabel = new Map();
    for (const variant of variants) {
      optionsByLabel.set(effortLabel(variant), variant);
    }
    const options = Array.from(optionsByLabel, ([label, row]) => ({ label, row }));
    const selectionKey = `${scope}:${key}`;
    const requested = selectedEfforts.get(selectionKey);
    const selected = optionsByLabel.get(requested) || options[0]?.row;
    if (!selected) continue;

    const selectedLabel = effortLabel(selected);
    selectedEfforts.set(selectionKey, selectedLabel);
    selectedRows.push({
      ...selected,
      __effortOptions: options.map((option) => option.label),
      __effortSelectionKey: selectionKey,
    });
  }
  return selectedRows;
}
