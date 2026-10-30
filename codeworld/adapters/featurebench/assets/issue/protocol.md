# FeatureBench Issue Author Protocol

`outbox/issue.json` is the only host-consumed authoring file. It contains exactly:

```json
{
  "task_statement": "...",
  "interface_contracts": [
    {"object_id": "...", "docstring": "..."}
  ]
}
```

The draft already contains every required `object_id`. Preserve each ID exactly once and replace
the empty strings with authored content. Intermediate edits do not need to validate; only a submit
is authoritative.

## Task statement

Follow the official FeatureBench authoring intent: write a generalized, short task statement that
summarizes the interfaces' core functionalities, main features and requirements, and key challenges
or considerations. Focus on repository-native goals rather than implementation-removal details.

## Interface contracts

For a function or method, document its main behavior, parameters, return behavior, and important
notes or exceptions supported by the source evidence. For a class, document its primary purpose,
main attributes, main methods, and a usage example when the evidence supports one. Preserve useful
existing documentation and terminology. Do not invent behavior.

Write only the inner docstring text: do not include triple-quote delimiters or Markdown code fences.
Do not copy implementation bodies. Do not mention benchmark construction, classifications, masking,
patches, or hidden tests.

## Evidence and validation

`input/top-interfaces.jsonl` is the authoritative interface manifest. Each record points to an
original, unmasked production file under `sources/`. Only files containing required interfaces are
present; no test files are available.

Call `submit_issue_file("outbox/issue.json")` when the file looks complete. A rejected submit writes
concise corrections under the returned feedback directory. Read those files, preserve valid content,
edit only what needs correction, and submit again. Finish only after the tool returns `accepted=true`.
