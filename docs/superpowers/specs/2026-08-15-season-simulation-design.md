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
    every tipster picks their level-0 best response (below)
    draw one result:  favourite wins with probability p_fav[t]
    every tipster whose tip was correct scores +1
    if it is a margin game, accumulate |actual - tip| for everyone

winner = highest score; ties broken by lowest accumulated margin error
```

**The leader tipping favourites is emergent, not imposed.** Whoever is genuinely
winning has terminal value 1.0 at delta 0, so deviating can only risk it and the DP
picks the favourite for them unprompted. There is no special case, which is what
lets the rule stay correct when "leading" and "winning" come apart.

The result is drawn **once per game and shared by every tipster**, which is what
couples them — they are tipping the same match.

Standings are re-read **every game**, so the always-favourite behaviour passes around
as the lead changes during the season. That is what the fixed start-of-season
assignment could not express.

### Each tipster's decision

A tipster believes everyone else tips favourites from here, so they win exactly when
they finish ahead of **every** opponent — counting a tie as a win only where their
own margin error is the lower one:

```
standing = [(points_j - points_i, who wins a tie) for every opponent j]

terminal(delta) = 0.0   if delta < gap for any j            (behind them)
                  0.0   if delta == gap and they win the tie
                  0.5   if delta == gap and the errors are equal
                  1.0   otherwise                            (clear of everyone)
```

Margin errors are those **accumulated so far in this simulation**, so the target
tightens or loosens as the margin games land.

**Points alone are not the state.** A tipster level with the top but holding the
worse margin error is not leading — a tie is a loss — and must keep chasing. An
earlier version keyed the decision on the deficit to the top score plus a single
countback flag against the current leaders, which made exactly this mistake: on
drawing level, the tipster was classed as a leader, stopped deviating, and settled
for a tie it would lose. On the live board that pinned NRL > AFL at 0%, reaching the
top score in 49.5% of seasons and never once alone.

That terminal feeds `solve_level0` over the *remaining* games, carrying the usual
`reluctance`, and the action is read at delta 0. So each tipster re-solves from the
live state every game, which is what "decide again once the result is known" means.

### Caching

The decision depends on `(game index, the multiset of (gap, tiebreak) over
opponents)`. The multiset is order-independent, so sorting canonicalises it. Distinct
keys number in the thousands across a whole run against 20,000 x 17 x 7 lookups, so
the DP still runs a few thousand times rather than millions. Measured cost of the
whole table at 20,000 seasons: about 3 seconds.

## Reporting

`--recommend` prints the simulated table, with the count and the standard error, and
one line reconciling it against the headline:

```
WHO WINS THE COMP  (simulated, 20000 seasons, +/- ~0.3%)
    Ryan Board       ..%
    Jake Turner      ..%   <-- you
    ...
    Your row is NOT the headline P(win comp) -- they are different models, and
    TWO things change at once. This row has you on the same level-0 rule as
    everyone else rather than your exact optimum, which costs you; but the
    rivals also change, gaining a moving leader role and losing their forced
    lockstep. Those pull opposite ways, so neither number bounds the other.
```

Neither direction can be assumed. On the live board the simulated row came out
*above* the headline, because the change to the rival model outweighed the loss of
my exact policy.

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
- Whoever is genuinely winning tips the favourite, without a special case.
- **A tipster level on points but losing the countback keeps deviating**, and ends up
  with a non-zero win probability rather than being pinned at 0.
- Of two rivals a point back off the same score, the one who would lose a tie must
  deviate at least as readily as the one who would win it.
- The same seed reproduces the same table exactly.
- Doubling the season count moves each probability by less than a few standard
  errors.

## Out of scope

- Changing the tip recommendation, which keeps the exact grouped solve.
- Any equilibrium solve. A tipster still assumes the rest of the field tips
  favourites when choosing their own action.
