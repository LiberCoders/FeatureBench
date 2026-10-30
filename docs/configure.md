# Configure

Edit an inactive run before resuming it:

```bash
fb configure outputs/build-YYYYMMDD-HHMMSS
```

The editor changes only run inputs:

- all stages: configuration file paths;
- Discover: repository source;
- Build: repository source, ref, and limit;
- Generate: environment source, environment ID, and limit.

Edit model, concurrency, timeouts, source endpoints, and Adapter settings directly in
the referenced YAML files.

Use the arrow keys to navigate, Enter to edit, and Ctrl+C to exit. Nothing is written
until Save is confirmed. A run that is currently active cannot be edited.

Resume after saving:

```bash
fb --resume outputs/build-YYYYMMDD-HHMMSS
```

Every execution stores the resolved settings under `.featurebench/invocations/`, so
direct YAML changes remain auditable. See [Resume](reference/resume.md) for work selection.
