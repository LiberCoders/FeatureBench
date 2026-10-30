Inspect /testbed and submit only the EnvironmentAnalysis required by the output schema.

Report every language needed to build and run one coherent unit-test suite for the repository's primary project,
along with that suite's build systems, package managers, and unit-test frameworks. Prefer the repository's
canonical top-level package suite. Omit independent optional clients, examples, documentation sites, generated
frontends, and separately released subprojects unless that primary suite actually builds or executes them. Put all
repository-evidenced runtime versions or constraints for the accepted suite into the single
runtime_requirement string; use null when none exists. It describes repository requirements, not a copy of a
base-image version.

When real unit tests exist, provide every real framework and repository-relative evidence proving its executable
tests and standard suite configuration. Do not design a Runner command or repeat a test-selection decision: the
Artifact Agent will inspect this evidence, while Runner collect will establish the sole authoritative test
inventory. When no real unit-test framework backed by executable unit tests exists, return empty test_frameworks,
null selected_base_image_name, and an evidence-based rejection_reason; the Harness derives the final eligibility
decision from those structured facts.
If that rejected repository also contains no evidence for any implementation language, return null
primary_language and empty languages rather than inventing `generic` or another placeholder. When language evidence
does exist, report it even though the repository is rejected.

Choose a base whose primary runtime family, version, capabilities, and GPU support are compatible with the
repository evidence. Match the base to primary_language. Do not reject a suite merely because it also needs a
secondary compiler or native-extension toolchain absent from the option's languages: when the option declares
build-toolchain, the Artifact Agent can install that evidenced secondary toolchain during Docker build. If real
unit tests exist but none of base_image_options is compatible,
keep the discovered test facts, return null selected_base_image_name, and set
rejection_reason to a concise explanation that no configured base image matches the evidenced requirements. For a
repository without real unit tests, also set rejection_reason to the evidence-based absence reason. For an accepted
repository, rejection_reason must be null.

Classify GPU requirements as follows: none means the accepted suite has no GPU dependency; optional means a
meaningful complete CPU test set exists even if separate GPU-only tests or acceleration exist; required means
no meaningful unit-test set can run on CPU. When the repository is accepted, optional GPU must describe the
complete CPU-runnable suite and select a CPU base, while required GPU must select a GPU-capable base. Cite GPU
evidence only for optional or required.

For an accepted testable repository, choose exactly one name from base_image_options. The Harness resolves its image
reference, clone plan, submodules, Git LFS, and ecosystem guidance; do not output those derived fields. Cite
only inspected repository-relative paths. Submit only the typed schema fields: do not propose activation,
Dockerfile, Runner, or an implementation plan, and do not add commentary outside gpu_reason or rejection_reason.
