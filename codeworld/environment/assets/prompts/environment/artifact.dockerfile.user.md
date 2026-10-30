Dockerfile:
- The first FROM must be selected_base.image exactly. The Harness inlines the managed base recipe, so every
  selected_base capability is already available; do not reinstall declared capabilities.
- Treat the accepted local unit-test suite as the dependency boundary. Prefer repository-evidenced lockfiles,
  canonical CI setup, and standard test workflows. Before selecting any optional dependency group, profile,
  feature, or extra, inspect its definition and include it only when repository evidence shows the accepted tests
  require it; a group's name alone is not evidence. Choose the smallest set that covers the accepted tests, and
  do not pull in dependencies belonging only to independent training, benchmark, documentation, example,
  external-service, or separately released subprojects. Respect committed locks using the package manager's
  actual semantics rather than adding unsupported strict flags.
- Put one line containing exactly {source_marker} in the final stage. The Harness replaces it with the pinned
  online clone and deterministic submodule/LFS setup. Do not COPY or ADD from the local build context, use
  /testbed before the marker, or mutate the pinned Git baseline.
- Materialize all third-party dependencies and required tools while build networking is available. Runtime uses
  execution_contract.resources.network and must not install, synchronize, update, or download dependencies; it may
  rebuild or regenerate only repository-owned artifacts offline when needed to consume the current /testbed source.
- Keep each network-backed source service in its own RUN instruction. Do not combine unrelated package indexes,
  installers, registries, or distribution downloads in one RUN; separate them so a failed BuildKit step identifies
  exactly one service for source fallback.
- Retain caches or reusable compiled state required for offline execution without expanding the accepted dependency
  boundary, but never make a stale build-time project artifact the Runner's source of truth.
