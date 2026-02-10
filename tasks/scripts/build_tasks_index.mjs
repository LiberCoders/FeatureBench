import fs from "node:fs";
import path from "node:path";

const root = process.cwd();
const tasksRoot = path.join(root, "tasks", "data");
const splitRoots = {
  lite: path.join(tasksRoot, "lite"),
  full: path.join(tasksRoot, "full"),
};

function walkProblemStatements(dir) {
  const out = [];
  const stack = [dir];
  while (stack.length > 0) {
    const current = stack.pop();
    if (!current || !fs.existsSync(current)) continue;
    for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
      const abs = path.join(current, entry.name);
      if (entry.isDirectory()) {
        stack.push(abs);
        continue;
      }
      if (entry.isFile() && entry.name === "problem_statement.md") {
        out.push(abs);
      }
    }
  }
  out.sort();
  return out;
}

function normalizeText(md) {
  return md
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/^\s{0,3}#{1,6}\s+/gm, "")
    .replace(/\bTask Statement:\s*/gi, "")
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\[(.*?)\]\((.*?)\)/g, "$1")
    .replace(/^\s*[-*+]\s+/gm, "")
    .replace(/^\s*Task\s+/i, "")
    .replace(/\s+/g, " ")
    .trim();
}

function previewMarkdown(md, maxChars = 360, maxLines = 8) {
  const rawLines = md.split(/\r?\n/).map((line) => line.replace(/\t/g, "  ").trimEnd());
  const lines = [];
  let totalChars = 0;

  for (const line of rawLines) {
    const isBlank = line.trim().length === 0;

    if (lines.length === 0 && isBlank) {
      continue;
    }

    if (isBlank && lines[lines.length - 1] === "") {
      continue;
    }

    const projected = totalChars + line.length + (lines.length > 0 ? 1 : 0);
    if (lines.length >= maxLines || projected > maxChars) {
      break;
    }

    lines.push(isBlank ? "" : line);
    totalChars = projected;
  }

  while (lines.length > 0 && lines[lines.length - 1] === "") {
    lines.pop();
  }

  return lines.join("\n");
}

function repoSlug(repoRef) {
  const repo = repoRef.split(".")[0] || repoRef;
  return repo.replace(/__/g, "/");
}

const items = [];
for (const [split, splitDir] of Object.entries(splitRoots)) {
  for (const file of walkProblemStatements(splitDir)) {
    const rel = path.relative(splitDir, file);
    const parts = rel.split(path.sep);
    if (parts.length < 3) {
      continue;
    }

    const repoRef = parts[0];
    const caseName = parts[1];
    const id = `${split}:${repoRef}/${caseName}`;
    const md = fs.readFileSync(file, "utf8");
    const normalized = normalizeText(md);
    const preview = normalized.slice(0, 420);
    const previewMd = previewMarkdown(md, 360, 8);

    items.push({
      id,
      split,
      repo_ref: repoRef,
      repo: repoSlug(repoRef),
      case: caseName,
      title: caseName,
      preview,
      preview_md: previewMd,
      statement_path: `./data/${split}/${parts.join("/")}`,
    });
  }
}

items.sort((a, b) => {
  if (a.split !== b.split) return a.split.localeCompare(b.split);
  if (a.repo !== b.repo) return a.repo.localeCompare(b.repo);
  return a.case.localeCompare(b.case);
});

const payload = {
  generated_at: new Date().toISOString(),
  counts: {
    lite: items.filter((x) => x.split === "lite").length,
    full: items.filter((x) => x.split === "full").length,
    total: items.length,
  },
  repos: Array.from(new Set(items.map((x) => x.repo))).sort(),
  items,
};

const outFile = path.join(tasksRoot, "tasks_index.json");
fs.writeFileSync(outFile, JSON.stringify(payload, null, 2) + "\n", "utf8");
console.log(`Wrote ${outFile} with ${payload.counts.total} tasks.`);
