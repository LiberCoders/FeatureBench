# Dataset V1.1 leaderboard data

`lite.json`, `full.json`, and `fast.json` contain one JSON object per evaluated effort. The `effort` value is free text; the frontend does not enforce a fixed list.

Rows with the same exact `agent` and `model` values are displayed as one leaderboard row. If that group contains more than one distinct `effort`, the Effort cell becomes a dropdown and the selected record controls the scores and date shown in the row.

Effort options follow their first appearance in the JSON file, and the first effort listed for an Agent+Model group is selected by default. Selecting another effort re-sorts and re-ranks the leaderboard using that effort's scores.

Example only (the numbers below are not benchmark results):

```json
[
  {
    "agent": "Codex",
    "model": "GPT-5.6",
    "effort": "low",
    "passed": 62.3,
    "resolved": 31.5,
    "agent_org": "OpenAI",
    "model_org": "OpenAI",
    "date": "2026-08-27",
    "filter_1": ["Close Scaffold", "Close Weights"],
    "filter_2": [
      "Model: GPT-5.6",
      "Scaffold: Codex",
      "Agent Org: OpenAI",
      "Model Org: OpenAI"
    ]
  },
  {
    "agent": "Codex",
    "model": "GPT-5.6",
    "effort": "xhigh",
    "passed": 71.8,
    "resolved": 42.0,
    "agent_org": "OpenAI",
    "model_org": "OpenAI",
    "date": "2026-08-27",
    "filter_1": ["Close Scaffold", "Close Weights"],
    "filter_2": [
      "Model: GPT-5.6",
      "Scaffold: Codex",
      "Agent Org: OpenAI",
      "Model Org: OpenAI"
    ]
  }
]
```

The V1.1 tab remains disabled while all three data files are empty. It becomes available automatically—and becomes the default version—when at least one V1.1 split contains a result. `?version=v1.0` can still be used to open the historical leaderboard directly.

`siteUrl` is not part of the V1.1 schema.
