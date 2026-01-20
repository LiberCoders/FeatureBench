# ACE-Bench Leaderboard

Static ACE-Bench leaderboard page.

## Data (backend)

Edit the leaderboard data in these files:

- [data/lite.json](data/lite.json) (Lite tab)
- [data/full.json](data/full.json) (Full tab)

To add a new entry, append one object (one row) to the corresponding file.

- You do NOT need to pre-sort; the frontend sorts automatically.

### Filtering fields

Filter option definitions live in [data/filter_options.json](data/filter_options.json):

- `allFilters`: shown in the “All Filters” dropdown
- `allTags`: shown in the “All Tags” dropdown (currently used as tag categories)

Each row in `lite.json` / `full.json` supports:

- `filter_1`: string or string[] (multi-select), values must come from `allFilters`
- `filter_2`: string or string[] (multi-select), values are free-form tags like `Model: xxx`, `Scaffold: yyy`, `Org: zzz`

Entry fields:

- `model` (string)
- `passed` (number, percent)
- `resolved` (number, percent)
- `org` (string | string[]) local logo image path(s), e.g. `./logos/openai.png`
- `date` (string, e.g. `YYYY-MM-DD`)
- `siteUrl` (string, URL; rendered as an external-link icon)

## UI (frontend)

The UI is implemented as a static site:

- [index.html](index.html) (markup)
- [styles.css](styles.css) (styles)
- [src/app.js](src/app.js) (ES module entry; imports the rest of `src/*.js`)

- Default sorting: `%Passed` (descending)
- User sorting: click the triangle button in `%Passed`, `%Resolved`, or `Date`
- Theme: Light/Dark toggle (remembers your choice)

### Top badges (above lite/full)

The badge links shown above the `lite/full` tabs are configured in:

- [data/top_badges.json](data/top_badges.json)

Format:

```json
{
	"items": [
		{ "label": "arXiv", "value": "25xx.xxxxx", "href": "https://arxiv.org/abs/...", "tone": "neutral" }
	]
}
```

Supported `tone` values: `neutral` | `info` | `success` | `danger` | `arxiv` | `github` | `huggingface`.

## Preview locally

Because the page loads JSON via `fetch`, you should serve it over HTTP.

```bash
python3 -m http.server 8000
```

Then open:

- http://localhost:8000/index.html
