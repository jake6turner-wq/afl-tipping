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

## Choosing the next tip from the simulation

The decision is read off the simulation rather than the exact solve, so the headline
and the table agree by construction.

Each season is drawn **once** and played **twice**: my next tip forced to the
favourite, then to the underdog, over identical results and identical margin draws.
The branch with the higher eventual win rate is the recommendation, and the table
shown is that branch's table — so my row in it *is* the headline number.

From the second game on I play the same level-0 rule as the field, so this is exact
one-step lookahead over a rollout policy, not a full optimum.

**The pairing does not reduce variance here — it slightly increases it.** The same
result that rewards tipping the favourite punishes tipping the dog, so the branches
come out mildly anti-correlated: measured on the live board, the paired standard
error on the edge is 0.0049 against 0.0043 for independent sampling. Pairing is kept
because it answers the actual counterfactual — in *this* season, which choice was
better — and `stderr_edge` is computed from the realised per-season differences, so
it reports the true uncertainty instead of understating it with an independence
formula.

Two warnings fire: when the edge is inside twice its standard error, and when the
exact grouped solve prefers the other tip.

`solve_joint` still runs. It supplies the cross-check, the contingency table, and
every sensitivity section, which stay on the exact grouped model and are therefore
relative comparisons rather than numbers comparable to the headline.

## The next deviation

Every season already records the index of the first game at which I take the
underdog, so reporting it costs nothing beyond a histogram.

It is deliberately reported as a **distribution over games with a "never" row**, not
as a single planned game. Whether I deviate at game `g` depends on the standings when
`g` arrives, which depend on results before it. An earlier version of this engine
printed a row of the policy table as though it were a sequence of future tips; that
was wrong, because every entry after the first deviation sits on a delta the tipster
has just left. There is no plan, only a rule.

Probabilities use `pct_mc`, so a unanimous outcome prints `>99.998%` rather than
claiming a certainty 60,000 draws cannot support.

## The contingency table

Driven by the same simulation, so no section of the report contradicts another.

Cell `(t, d)` asks what to tip at game `t` with my score `d` off the all-favourites
baseline and the rivals on their level path. That is exactly the season problem
starting at game `t` with my points shifted by `d`, so it reuses
`simulate_branches` on `games[t:]` with no new machinery.

**Cell (0, 0) is the headline object itself**, not a re-estimate. It is the same
quantity the report leads with, and a second smaller sample of it could land the
other side of a close call and contradict the recommendation — which is precisely
the bug this replaced. Passing the headline in guarantees agreement.

There are about 15 cells, so it runs at its own lower season count
(`--contingency-seasons`, default 6000). At that size the edge error is around one
percentage point, which cannot resolve every cell, so cells whose branches sit
within two standard errors are printed `too close to call` rather than given a
winner the sample does not support.

`BASELINE COMPARISON`, `CHASER MODEL SENSITIVITY`, `DEVIG SENSITIVITY` and
`MARGIN TIP` still come from the exact grouped solve and are labelled
`[exact grouped solve -- relative only]` in their headers, so the boundary between
the two models is visible at the point of reading rather than buried in a footnote.

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

`--sim-seasons` sets the count, default 60000 (raised from the initial 20000 once the runtime proved cheap).

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
