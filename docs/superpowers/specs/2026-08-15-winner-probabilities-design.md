# Win probability for every tipster — design

Date: 2026-08-15

## Objective

Add to `--recommend` a table giving each tipster's probability of finishing first,
under one shared world in which everyone plays their best response from here on.

## Interpretation

"Everyone plays optimally" is only well defined once you fix what each player
believes. This uses the engine's existing model rather than inventing a second one:

- **Me:** exact backward induction over `(game, my delta, each rival group's delta)`.
- **The points leader:** tips favourites, per the standing instruction.
- **Chasers:** their own level-0 best response, including the price-scaled
  reluctance from `[[2026-08-14-deviation-reluctance-design]]`.

Every row is therefore a best response *by that player's own lights*. Because it is
a single shared world, the column is a genuine probability distribution and sums to
1. That is the point of choosing this reading over seven separate counterfactuals:
those would answer "how would each rival do with this tool" and would not sum to
anything meaningful.

Two consequences are stated in the block's footnote rather than left implicit:

- The leader's number is his probability **under the always-favourite instruction**,
  not under a free optimisation. It is not a claim about how well he could do.
- Rivals do not respond to me. They play the field, exactly as everywhere else in
  the report.

## Why a forward pass reconciles

Every tipster tips the same games. If two of them both take the dog, they both move
the same way and the gap between them is unchanged; the gap moves only when they
disagree, and then by exactly 1. **Any pair's gap therefore changes by at most 1 per
game.**

This is the same invariant the original design relies on, and it was verified here
rather than assumed: `solve_joint`'s elimination prune (`worst - remaining > 0`)
rests on it, and removing the prune entirely changes `p_win` by 0.0000000000. So the
backward solve and a forward propagation of the same policies must agree, which
makes "my row equals the headline" a real test rather than a tautology.

## Component 1: countback learns who wins

`CountbackModel` answers only "do *I* beat subset S". A tie among rivals that does
not involve me is unrepresentable, so it gains:

```python
def winner_probs(self, members: Tuple[int, ...]) -> List[float]:
    """P(each member holds the lowest cumulative margin error).
    Unified indexing: 0 = me, 1..n = rivals."""
```

`_simulate` already computes every tipster's error on every draw and currently
discards all but the comparison against mine. It will retain them in a flat
`array('d')` of `n_sims * (n_rivals + 1)` doubles — 5.6 MB at the default 100k sims,
with no per-float object overhead. `winner_probs` scans for the lowest error among
`members` on each draw and is cached per member set; the forward pass produces only
a handful of distinct tied sets.

Storing raw errors rather than a precomputed ranking avoids sorting all seven
tipsters on every draw when almost every tie involves two or three.

## Component 2: forward propagation

```python
def winner_probabilities(me, rivals, p_fav, groups, countback, solution)
        -> List[Tuple[str, float]]
```

Starts at `(delta_me=0, group_deltas=(0, ...))` holding mass 1.0, and for each game
takes my action from `solution.action_at`, each group's from its own policy, and
splits the mass by `p_fav[t]`. State is a sparse dict keyed by
`(delta_me, group_deltas)`.

The reachable space is small — measured at **26 states at peak** across the live
18-game fixture, with terminal mass summing to 1.0 to twelve places — because the
leader group never branches and the chasers only deviate late. No pruning,
approximation, or sampling is required.

At the terminal, each state's final score is `points + delta` (the `F` term cancels,
as everywhere else in the engine). Whoever holds the maximum is tied for first, and
that state's mass is split across them by `winner_probs`. Results are accumulated
per tipster and returned in descending probability, me included.

## Component 3: reporting

A block in `--recommend`, after the countback safety table:

```
------------------------------------------------------------------------------
WHO WINS THE COMP  (everyone playing their best response, one shared world)
------------------------------------------------------------------------------
    Jake Turner        38.30%   <-- you
    Ryan Board         ...
    ...
    ---------------------------
    total             100.00%
```

`recommendation.csv` gains one `winner,<name>,<probability>` row per tipster.

## Testing

- The returned probabilities sum to 1.
- **My entry equals `solution.p_win`** to twelve places — the reconciliation above.
- Every probability lies in `[0, 1]` and every tipster appears exactly once.
- A rival leading by more than the number of remaining games wins with probability 1.
- With no games remaining, all mass goes to the current leader, or splits across a
  tie by countback.
- `winner_probs` over a single member returns `[1.0]`, and over a set sums to 1.

## Out of scope

- Letting the leader optimise instead of tipping favourites.
- Per-person counterfactuals ("how would they do with this tool").
- Any equilibrium solve. Rivals still do not respond to me.
