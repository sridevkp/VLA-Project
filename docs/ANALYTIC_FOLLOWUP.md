# Isolated analytic follow-up

Branch: work/analytic-risk-followup. This worktree snapshots the current world-model,
RiskGate and core interface context from the accepted local work. It does not modify
the frozen VLA source/data archives, registration, gate, scene splits or deadlines.

Geometry now rejects nonfinite/misshaped inputs, negative radii and reversed box bounds.
World correction rejects duplicate object identities rather than silently collapsing them.
Existing hard-veto utility and bounded recovery tests remain in force. Tool-sphere geometry
is not full-arm collision checking; grasp probabilities and utility calibration remain pending.
