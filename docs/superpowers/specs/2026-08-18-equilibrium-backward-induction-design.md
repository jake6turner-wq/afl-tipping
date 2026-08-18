# Simulated backward induction — an approximate equilibrium

Date: 2026-08-18

## Objective

Replace the one-step lookahead over a level-0 rollout with a policy derived by
**backward induction over simulated seasons**, so that every tipster is responding
to how the others actually play rather than to the fiction that they tip favourites.

## Why not exact

Each game the field splits into favourite-tippers and dog-tippers and one side
scores, so any 0/1 gain vector is reachable. After `G` games the joint state is any
vector in `{0..G}^n`: **10^10 states for 10 players over 9 games**, with `2^10 = 1024`
simultaneous action profiles to solve at each node. Exact backward induction with a
Nash solve per node is out by orders of magnitude, and a general-sum equilibrium
would be neither unique nor guaranteed reachable by iteration.

## What is computed instead

Backward induction on **sampled trajectories**, over the state abstraction the engine
already uses.

A tipster's decision state is `(game index, the multiset of (points gap, who wins a
tie) over the opponents)` — the same key `_decision_cache` uses today. It is
order-independent and compact, and it is shared by every tipster, so one policy table
serves all ten and pools their samples.

```
for t from the LAST game back to the first:
    repeat a few sweeps:
        for many sampled seasons:
            play games 0..t-1     with the current policy, plus exploration
            at game t             pick one focus tipster; they take a random
                                  action, everyone else uses the current policy
            play games t+1..end   with the ALREADY FINALISED policy
            record (focus's state at t, action taken, whether they finished first)
        set policy[t][state] = whichever action won more often
```

Going backwards is what makes this well-founded: when game `t` is decided, every
later game already has its settled policy, so the value of an action at `t` is
measured against optimal continuation rather than against a rollout heuristic.

The sweeps at a fixed `t` are iterated best response *within the stage*: sweep 1
measures against opponents playing the incoming policy, sweep 2 against the policy
sweep 1 produced, and so on. The stage settles to an approximate equilibrium before
the induction moves to `t-1`.

## Honest limits

- **It is an approximate equilibrium, not a proven one.** The stage sweeps may cycle
  rather than settle; the number of sweeps that actually changed nothing is reported.
- **The state abstraction is a modelling choice.** Two genuinely different joint
  positions that share a gap-and-tiebreak signature are treated as one state.
- **Sampling error.** Each `(state, action)` pair is a Monte Carlo mean; states
  reached in few seasons carry wide errors, and the count per state is reported.
- **A symmetric policy.** All ten tipsters share one rule keyed by their own
  situation. That is what makes it tractable and sample-efficient, but it cannot
  express a rival who is idiosyncratically reckless.
- Unvisited states fall back to the existing level-0 decision rather than guessing.
- **A state must clear `min_visits` samples per arm before it is kept.** The first
  build without this bar learned 3292 states, some decided on a *single* sampled
  season, and flipped the scenario recommendation on that basis. Writing a coin flip
  into the table dresses noise up as knowledge; a thin state is now left on the
  level-0 rule and counted separately.
- The fallback count is reported. If it dwarfs the learned states, most decisions are
  still level-0 and the run should not be leaned on.

## Interface

`--equilibrium` is opt-in. The default `--recommend` keeps the current fast engine at
about 23 seconds; the new solver takes minutes.

When enabled, the learned policy replaces `_decision_cache` in the simulation, so the
headline, the win-probability table, the next-deviation block and the contingency
table all flow from it unchanged — they already take the decision rule as an argument.

Both engines' recommendations are printed and a `WARNING` fires when they disagree,
matching the existing exact-solve cross-check.

`--equilibrium-seasons` and `--equilibrium-sweeps` control the budget.

## Testing

- A tipster who cannot be caught is given the favourite at every visited state.
- With one game left the learned policy matches exact reasoning: take the dog only
  when the tie or the deficit demands it.
- Learned policies are deterministic under a fixed seed.
- The solver reports how many states it learned, the minimum sample count behind any
  of them, and whether the final sweep changed any action.
- Falling back on an unseen state returns the level-0 action rather than raising.
- Against a decided race the equilibrium and the current engine agree.
