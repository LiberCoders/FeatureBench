# FeatureBench Mask Repair Protocol

`outbox/proposal.json` is the only host-consumed Agent file. It contains exactly:

- `based_on_attempt`: the current feedback `Attempt` whose Mask Plan receives this increment. Copy
  the refreshed value from `outbox/proposal.json`; do not copy a prior proposal's base attempt.
- `must_restore_object_ids`: currently masked objects that must leave the Mask Plan.
- `must_mask_object_ids`: currently unmasked eligible objects that must enter the Mask Plan.
- `rationale`: a short audit explanation.

The two object lists are incremental relative to `based_on_attempt`. The host validates atomic
bindings, protection, structural nesting, dependency closure, and real tests. Top/Specific
classification and other closure changes remain host-managed.

Before adding a mask, search `catalog/objects.jsonl` for records with `eligible_for_mask=true` and
`protected=false`, and exclude IDs listed in the current feedback's `learned-protected.jsonl`.
`estimated_seed_affected_points` and `estimated_p2p_affected_points` summarize trace evidence so a
small seed-relevant, low-P2P increment can be chosen without loading test-ID lists. They are estimates;
only `submit_repair_file` decides the real outcome.
