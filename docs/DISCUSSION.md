# Discussion

## VALIDATION interpretation

**Observed.** Frozen WorldMind achieved 14/18 successes against the scripted
baseline's 15/18. It agreed with the heuristic arm on all matched identities.
The predictive hard veto activated on about 0.03–0.05% of evaluated candidates.
The equal-weight predictive ranking ablation reached 15/18, with two discordant
identities against baseline. These small paired populations describe observed
outcomes without establishing superiority or equivalence.

**Inferred from control flow.** The major arm-5 failures happened before the
predictive decision boundary. The expert IK solver could not produce a waypoint
command, so neither the pool nor its predicted placement existed. This is a
feasibility result under the configured scripted pathway and bounds. Global
kinematic reachability remains untested. The goal-heavy RETREAT veto shows
a separate case where the predictive channel affected execution.

**Prediction fidelity and cost.** Object-position RMSE at one tick was roughly
65 mm, versus approximately 13 mm per side of grasp clearance. These different
measurement populations give scale context and cannot calibrate the probability
of a missed grasp. The model's nominal tracking, lack of contact dynamics and
no inferred grasp limit fidelity. Missing long executable pairs prevent a
matched 50-tick accuracy claim. Decision p95 increased from 18.8 ms with the
heuristic arm to 1,093.7 ms with frozen WorldMind.

**Safety and external validity.** The frozen veto covers tool and scene geometry
in simulation. Zero recorded nonrobot contacts and a saturated displacement
proxy provide no evidence of collision reduction. Finger coordinates can stray
from motor target bounds by about 10⁻⁶ m, reported as simulator numerical noise
under the logging-only decision. Command limits and task criteria retain their
original meanings. Full-body collision and physical-robot safety remain unmeasured.

**VLA limitation.** The VLA-only TRAIN promotion result remains 0/24. Arms
3–5 used the scripted proposer after the failed gate. This
experiment does not establish improvement over a competent VLA. The arm-3
versus arm-4 contrast differs in both prediction availability and selection
policy, limiting attribution to prediction alone.
[VALIDATION observations](RESULTS.md)

## TEST independent interpretation

Frozen WorldMind achieved 23/30 versus the scripted baseline's 24/30.
It succeeded on one identity the baseline missed and failed on two identities
the baseline passed. The scene-level exact sign test gave p = 1.00 across ten
paired scene units. No statistically significant improvement was detected.
Statistical equivalence has not been established. The three 23/30 totals for
arms 3, 4 and 5 conceal different paired outcomes.

The TEST hard veto rejected 196 of 347,616 evaluated candidates (0.0564%).
One TEST LIFT exception recovered on the first observed tick. The three
successful VALIDATION LIFT recovery events are a separate observation set.
No collision reduction was measured. Improvement over a competent VLA was
not established. The VLA-only TRAIN result remains 0/24.
[TEST observations](RESULTS.md#test-independent-evaluation)
