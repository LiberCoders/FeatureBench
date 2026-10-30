Runner:
- Return one complete Bash-compatible runtime adapter implementing runner_contract within every limit in
  execution_contract. It may delegate to an installed language but must use the real installed test framework
  and framework-native machine-readable evidence.
- The Runner is the environment's verifier boundary: collect defines the authoritative test inventory, run must
  execute both singleton and batched selections from that inventory, and the Harness calls run over every collected
  path group in bounded deterministic chunks to retain only test points with explicit passed outcomes. If the real
  framework report omits a requested ID, return that ID conservatively as error; never fabricate another outcome.
- Treat requested IDs as unbounded structured input. Keep them in the supplied request file or another temporary
  selection file when delegating to a framework process; argv and environment variables may carry bounded file
  paths and fixed framework options, never the serialized ID collection itself.
- Apply run selection through the framework's native bounded selection mechanism before it imports or evaluates
  unrelated test sources whenever that framework supports pre-collection selection. Filtering an unscoped complete
  collection afterward is not equivalent because unrelated collection failures can corrupt the selected run. If
  native framework IDs are nondeterministic, collect may expose stable Runner IDs only when run can restrict native
  resolution to the requested IDs' source paths and, inside the same framework execution invocation, map every
  requested ID one-to-one to a current native item, select those mapped items, and map machine reports back without
  fabricating outcomes. Never resolve an unstable native ID in one process and replay its string in another.
- Derive the real framework entry point and one coherent local unit-test suite from repository CI, configuration,
  and test-layout evidence. A stable suite-level path or canonical command may exclude independently configured
  integration, end-to-end, external-service, example, and optional-GPU suites. It must include every test inside
  the accepted unit suite and must never select individual tests based on observed outcomes. collect is the sole
  authoritative inventory for that boundary: reporter and transport options may obtain machine-readable results,
  but must not add further dynamic selection, exclusion, or repository-observable presentation flags. If a tool
  such as tox or nox provisions environments, provision it during image build and invoke the resulting installed
  test environment directly at runtime.
- For optional GPU on a CPU runtime, exclude only tests explicitly evidenced as GPU-only. Never narrow ordinary
  tests or convert missing execution evidence into success.