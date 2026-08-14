# Ungrouped season simulation — design

Date: 2026-08-15

## Objective

Replace the grouped forward pass behind `WHO WINS THE COMP` with a Monte Carlo that
plays the remaining season out game by game, every tipster tracked individually and
re-deciding after each result.

## Why the grouped version had to go

Rivals with identical policies were collapsed onto one delta dimension. That assumed
they tip identically for the rest of the season, which froze their relative order and
gave Markash, DeanLFC and Mikefooty exactly 0% — not because they cannot win, but
because NRL > AFL shared their policy and started a point ahead. A tipster who
decides to chase a run of upsets is making their own choice, and the model should let
them.

## What stays grouped, and why

`solve_joint` — the exact backward induction behind the tip recommendation — keeps
using groups. Ungrouped, its state is seven independent deltas: roughly `41^7`
nodes against the ~26 the grouped forward pass actually reaches. That is not a
tuning problem, it is intractable.

**Consequence:** the headline `P(win comp)` and my row in the simulated table come
from two different models and will not agree. Both are printed, labelled, with the
reason stated inline. The headline stays exact, because the tip recommendation rests
on it; the table is a sampled estimate.

## The simulation

One season, repeated `n_seasons` times:

```
scores[i] = points[i]          errors[i] = margin_error[i]        (i = 0 is me)

for each remaining game, in lock order:
    leaders = every i holding the current maximum score
    for each tipster i:
        i in leaders          -> tip the favourite
        otherwise             -> their level-0 best response (below)
    draw one result:  favourite wins with probability p_fav[t]
    every tipster whose tip was correct scores +1
    if it is a margin game, accumulate |actual - tip| for everyone

winner = highest score; ties broken by lowest accumulated margin error
```

The result is drawn **once per game and shared by every tipster**, which is what
couples them — they are tipping the same match.

The leader is recomputed **every game**, so the always-favourite role passes around
as the lead changes during the season. That is the behaviour the fixed
start-of-season assignment could not express.

### Each tipster's decision

A non-leader believes everyone else tips favourites from here. Their final position
relative to the field then depends only on their deficit to the current leaders:

```
need      = (current top score) - (my score)
tied set  = the current leaders
terminal(delta) = 1.0            if delta >  need      (outright)
                  countback      if delta == need      (tie for first)
                  0.0            otherwise
```

`countback` is their belief about the tiebreak, from margin errors **as accumulated
so far in this simulation** — the existing hard 1 / 0.5 / 0 rule, multiplied across
the tied set, exactly as `rival_terminal` already does.

That terminal feeds `solve_level0` over the *remaining* games, carrying the usual
`reluctance`, and the action is read at delta 0. So each tipster re-solves from the
live state every game, which is what "decide again once the result is known" means.

### Caching

Every non-leader at a given game shares the same tied set, so the decision depends
only on `(game index, need, countback belief)`. The number of distinct keys is in the
hundreds across a whole run, against 20,000 x 18 x 7 lookups, so the DP is solved a
few hundred times rather than millions.

## Reporting

`--recommend` prints the simulated table, with the count and the standard error, and
one line reconciling it against the headline:

```
WHO WINS THE COMP  (simulated, 20000 seasons, +/- ~0.3%)
    Ryan Board       ..%
    Jake Turner      ..%   <-- you
    ...
    Your headline P(win comp) above is the EXACT solve of your optimal policy.
    This row has you playing the same level-0 rule as everyone else, so it is
    lower. The gap is what your optimal policy is worth.
```

`recommendation.csv` gains `winner,<name>,<probability>` rows plus a
`winner,_seasons,<n>` row recording the sample size.

`--sim-seasons` sets the count, default 20000.

## Removed

`winner_probabilities`, `shared_policy_blockers` and `CountbackModel.winner_probs`
go, with their tests. They exist only to serve the grouped table, and leaving them
would be dead code with no caller.

`CountbackModel` itself stays — `solve_joint` and the countback safety table still
use `subset_prob` and `pairwise`.

## Testing

- Probabilities sum to 1 and every tipster appears exactly once.
- A tipster who cannot be caught wins every season.
- Rivals on equal points with equal margin errors get equal probability, within
  sampling error, which the grouped model could not produce.
- **A rival who shared a policy group under the old model now has a non-zero
  probability** — the point of the change.
- The current leader always tips the favourite; a rigged fixture where the leader
  would otherwise deviate confirms the rule fires.
- The same seed reproduces the same table exactly.
- Doubling the season count moves each probability by less than a few standard
  errors.

## Out of scope

- Changing the tip recommendation, which keeps the exact grouped solve.
- Any equilibrium solve. A tipster still assumes the rest of the field tips
  favourites when choosing their own action.
