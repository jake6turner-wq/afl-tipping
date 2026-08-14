# Price-scaled deviation reluctance — design

Date: 2026-08-14

## Objective

Make modelled rivals deviate readily in close games and rarely in lopsided ones,
and stop the report asserting leaderboard facts it has not checked when running a
scenario set.

## The problem

A level-0 chaser's terminal reward is a step function: they either finish first or
they do not. Once the target binds — they need +2 with few games left — they must
take dogs, and taking a dog at 85% is no worse than at 65%, because both are
*necessary*. The DP only discriminates on price when it has slack.

Observed on the R23/R24 fixture, at delta 0:

```
R23G1-G9   p_fav 71-96%   fav        (chaser waits out the whole round)
R24G1      p_fav 58.5%    DEVIATE
R24G8      p_fav 85.3%    DEVIATE    <- as willing as the 65% games
```

This is not a coding error. It is the model being risk-neutral over a binary
target. Fixing it means adding a behavioural assumption.

## Mechanism

A **reluctance margin** the deviation edge must clear before a rival will take the
dog, scaled by the favourite's price:

```
penalty(p) = reluctance * max(0, p - 0.5)
take the dog iff  v_dog > v_fav + penalty(p)
```

`p` is the favourite's win probability, so `penalty` is zero at a coin flip and
maximal at a certainty. This is a smooth reluctance, not a cutoff: there is no
price at which deviation becomes impossible, only progressively less attractive.

**The stored value is the true value of the chosen action, never the penalised
one.** The penalty exists to select the action; propagating it into the value
function would make `values` a fictional score rather than the actual win
probability under the behavioural policy. Every consumer of `values` — the
countback terminal, the joint solve — depends on it being a real probability.

## Default

`RELUCTANCE = 0.10`, calibrated by sweeping against a chaser's policy on the live
fixture:

| `k` | deviates at | skips |
|---|---|---|
| 0.00 (today) | 58, 65, 66, 68, 69, 76, **85** % | 96% |
| **0.10** | 58, 65, 66, 69, 76 % | **85, 85, 96** % |
| 0.20 | 58, 65, 69 % | 66, 76, 85, 85, 96 % |

At `k = 0.10` a chaser needs roughly 3.5 points of extra win probability before
backing a 19.5-point underdog, and 1.5 points at 13.5 points of line. Every game
at 85% or shorter drops out; the 58–76% band stays in.

Like `TAU_TIP`, this is **ASSUMED, not fitted**. It is printed in the
`ASSUMPTIONS` block and exposed as `--reluctance` for sensitivity testing.

### Known limitation

The penalty interacts with position in the fixture, so it is not a clean per-game
price filter. At `k = 0.10` a 76% game can survive while a 68% game drops out,
because a later game's value depends on which deviations remain available after
it. The heavy-favourite games drop out reliably; ordering within the middle band
does not. This is inherent to applying a per-step penalty inside a sequential DP
and is documented rather than worked around.

## Scope

Rivals only. `solve_level0` gains a `reluctance` parameter defaulting to `0.0`,
so existing behaviour and every existing test are unchanged unless the caller opts
in. `build_rival_groups` passes the configured value through to each rival's DP.

`solve_joint` is untouched. My own policy stays the exact optimum against the new
rival model, so the reported `P(win)` remains a true optimum rather than a
constrained one.

## Report correctness under scenario sets

The WHY narrative currently hardcodes two claims and checks neither:

- *"You trail X by N point(s)"* — prints a negative gap when I lead.
- *"and hold the lowest margin error in the field"* — asserted unconditionally.

Both are true of the real post-R22 board and false in plenty of editable
scenarios. They become derived: lead / trail / tie chosen from the actual gap, and
the margin-error claim stated only when it holds, replaced by the true standing
otherwise.

## Testing

- A rival with a high reluctance deviates strictly less often than one with none.
- `reluctance=0.0` reproduces today's policy exactly.
- The penalty is zero at `p = 0.5`, so a coin-flip game is unaffected by any `k`.
- Stored values remain valid probabilities in `[0, 1]` under a non-zero reluctance.
- The WHY text says "lead" when ahead and does not claim the lowest margin error
  when a rival holds it.

## Out of scope

- Fitting `reluctance` from past rounds. It is assumed, like `tau`.
- Applying reluctance to my own policy.
- Any change to the devig, the countback, or the joint solve.
