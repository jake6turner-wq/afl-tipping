# Probabilistic rival deviation — design

Date: 2026-08-20

## Objective

Stop modelled rivals from converting a knife-edge preference into a certain,
field-wide action. Rivals should take the dog *some percentage of the time*, and
rivals in the same position should be able to tip differently from each other.

## The problem

`solve_level0` selects actions with a hard argmax
([tipping.py:355](../../../tipping.py#L355)):

```python
if v_dog > v_fav + floor + 1e-12:
    values[t][i], policy[t][i] = v_dog, "D"
```

Rivals are then grouped by identical policy signature, so a group moves as a bloc.
On the R24 board every chaser shares one policy, and their deviations rest on
margins far too thin to justify certainty. A chaser at delta 0:

```
game       dog   action     edge
R24G1    31.1%      fav   -3.13%
R24G3    38.4%      DOG   +1.61%     <- five rivals, all certainly on Carlton
R24G7    42.6%      DOG  +13.94%
R24G8    42.3%      DOG   +2.54%
```

A **1.61-point** edge produces five rivals on Carlton with probability 1. Any
analysis that leans on "the chasers will be on this dog too, so taking it covers
me" is resting on that, and it should not be.

This is not a coding error, and it is not fixed by tuning `reluctance`. It is the
model being infinitely confident about a near-indifference. Fixing it means adding
a behavioural assumption, as `reluctance` and `tau` already did.

Note the ordering the current model already gets right: the late deviations carry
the large edges (G7 at +13.94%) and the early one is marginal (G3 at +1.61%). Any
softening therefore erases the early bloc deviation while leaving the late ones
standing. That emergent behaviour is the goal; it is not encoded directly.

## Mechanism

A **logistic on the existing edge**, with a temperature expressed in the same units
as the values themselves (fractions of win probability):

```
edge = v_dog - v_fav - penalty(p)          # penalty is the existing reluctance
q    = 1 / (1 + exp(-edge / temperature))  # P(tipster takes the dog)
v    = q * v_dog + (1 - q) * v_fav
```

The value is the noise-weighted mixture, not the max — soft value iteration. A
rival's stored value therefore reflects the noisy player they actually are, rather
than an optimal player they will not be.

This preserves the invariant the reluctance design set: **the stored value is a
true win probability.** It is now the win probability under the noisy behavioural
policy rather than under an optimal one, which is a more accurate number, not a
less real one. The countback terminal and every downstream consumer still read a
genuine probability in `[0, 1]`.

`temperature = 0.0` recovers the hard argmax exactly and is the default for every
existing caller.

### Dead-state guard

When a tipster is mathematically eliminated, `v_fav = v_dog = 0`, the edge is zero
and the logistic returns 50% — a dead rival coin-flipping every game. This cannot
affect my win probability (a rival who cannot win cannot influence my finishing
position) but it makes `WHO WINS THE COMP` and the deviation diagnostics look
deranged. When both branches are zero, `q = 0`.

## Default

`RIVAL_NOISE = 0.05`, expressed as a fraction of win probability, so `0.05` is 5
percentage points and `--rival-noise 0.05` is the `T = 5%` column below. A rival needs roughly 5.5 points
of edge before they are 75% likely to take the dog. Calibrated against a chaser's
own edges on the live fixture:

| game | dog | edge | T=2% | **T=5%** | T=8% | T=15% |
|---|---|---|---|---|---|---|
| R24G1 | 31.1% | −3.13% | 17% | **35%** | 40% | 45% |
| R24G3 | 38.4% | +1.61% | 69% | **58%** | 55% | 53% |
| R24G5 | 3.6% | −19.65% | 0% | **2%** | 8% | 21% |
| R24G7 | 42.6% | +13.94% | 100% | **94%** | 85% | 72% |
| R24G8 | 42.3% | +2.54% | 78% | **62%** | 58% | 54% |

At `T = 0.05` the Carlton call becomes a genuine coin flip (58%), the Essendon
deviation stays near-certain (94%), and heavy longshots stay near zero (Richmond
at 2%). At `T = 0.15` a rival backs a 3.6% shot one time in five, which is not
credible — the tails are too fat above roughly 10%.

Like `tau` and `reluctance`, this is **ASSUMED, not fitted**. It is printed in the
`ASSUMPTIONS` block and exposed as `--rival-noise` for sensitivity testing by hand.

## Scope

**The season simulation only.** The exact grouped solve stays deterministic.

This is a deliberate tractability boundary, not an oversight. The grouped solve
collapses rivals sharing a policy into one shared delta dimension. Independent
mixing destroys that: rivals in a group would hold different deltas after the
first coin flip, so each would need its own dimension — roughly `41^7` states on
the current board against the ~26 the grouped version reaches. The simulation
flips an independent coin per rival for free.

The exact solve therefore keeps its existing documented role: a cross-check from a
different model, read for relative comparisons. The README already frames it that
way, so no new caveat is needed — but the divergence between the two models will
widen, and the cross-check `WARNING` may fire more often. That is informative
rather than a defect: it now fires when the recommendation depends on whether
rivals are treated as a certain bloc.

**Rivals only.** My own policy stays deterministic, so the recommendation is a best
response to a noisy field rather than itself being noisy. Same split `reluctance`
made.

## Implementation notes

### Noise draws must be pre-drawn per season

`_draw_season` pre-draws results and margins so both branches replay identical
luck: *"the only difference between the two branches is my forced first tip, not
the luck."* Rival noise is luck. It joins the pre-draw as a uniform per
`(game, rival)`, compared in each branch against whatever `q` that branch's
standings produce — same uniform, possibly different threshold. Common random
numbers.

Drawing inside `_play_season` instead would decouple the branches, inflate
`stderr_edge`, and make the `too close to call` flags meaningless.

### The decision cache still caches

`_decision_cache` returns a probability rather than an action string. The
probability is deterministic given `(t, standing)`, so the cache stays as
effective as it is today; only the draw moves to the caller.

`_actions` gains the per-game noise draws and the identity of the tipster, so
index 0 (me) bypasses the draw.

### `--equilibrium`

The equilibrium learner shares the simulation machinery and inherits noisy rivals
by construction. This is desirable — learning a best response against a noisy
field beats learning against a deterministic bloc. In scope to the extent of
verifying the path still runs and still reports sane diagnostics (states learned,
thinnest sample, fallback rate, non-convergence warning).

## Testing

- `temperature = 0.0` reproduces today's policy exactly; all 125 existing tests
  pass unchanged.
- `q` is monotone increasing in the edge.
- `q = 0.5` when the edge is exactly zero and the state is live.
- A dead state (`v_fav = v_dog = 0`) returns `q = 0`, not `0.5`.
- Stored values remain in `[0, 1]` at every temperature tested.
- Rivals sharing a policy take **different** actions within a single simulated
  season — the bloc is genuinely broken, not merely softened in the table.
- Both branches of a paired season observe identical rival noise draws.
- Raising the temperature widens the spread of `WHO WINS THE COMP`.
- The `--equilibrium` path runs to completion with noise active.

## Out of scope

- Fitting the temperature from observed rival behaviour. It is assumed, like `tau`
  and `reluctance`.
- Per-rival heterogeneous risk appetite (each rival drawing its own `reluctance`).
  Considered and set aside; noise on a shared rule is the smaller assumption and
  already breaks the bloc.
- An explicit "hold your deviation for the best remaining price" term. The DP is
  already full backward induction over all remaining games and prices this
  correctly; the observed early-deviation problem is knife-edge selection, not
  myopia.
- A noise sensitivity sweep block in the report. Sweep by hand with
  `--rival-noise`, as with `--reluctance`.
- Any change to the devig, the countback, the exact grouped solve, or my own
  policy.
