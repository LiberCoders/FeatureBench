# Pull

Pre-pull the Docker images used by a FeatureBench dataset split.

```bash
fb pull --mode lite
fb pull --mode fast
fb pull --mode full
```

`full` is the default. Use a text file to pull a custom image list:

```bash
fb pull --mode images.txt
```

Each non-empty line must contain one image reference. Lines beginning with `#` are
ignored.

Pull multiple images concurrently when the Docker registry and host can support it:

```bash
fb pull --mode full --n-concurrent 4
```
