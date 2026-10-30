Work only on the external environment artifacts represented by the current structured output schema for the
repository mounted at /testbed. The output schema is authoritative for both field presence and requiredness. Submit
every required field; omit an optional consistency field unless the direct repair actually requires changing it.

Dockerfile, activation, Runner, and FeatureBench data factory metadata stay on the host. Never embed them in the image, create
/opt/featurebench, install an Agent, or declare credentials. Preserve one coherent repository-evidenced local unit-test
suite and apply only the supplied ecosystem guidance for the artifact fields present in this run.

The pinned /testbed checkout is the sole source of project-under-test code. After repository source changes, tests
must load those changes directly or rebuild or regenerate project artifacts from the current checkout; never let a
detached build-time project copy shadow it. Dependencies may be frozen; repository source must remain live.
