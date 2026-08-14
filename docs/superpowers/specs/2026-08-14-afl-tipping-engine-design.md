# AFL tipping-comp optimal strategy engine — design

Date: 2026-08-14

## Objective

Answer one question repeatedly over R23–R24: **what do I tip in the next game that
locks, and what is my `P(finish first)` under each option?** Strictly `P(first)` —
no second prize, so expected rank is irrelevant.

## Confirmed competition rules

| Item | Answer | Source |
|---|---|---|
| Tiebreak for first | Lowest cumulative margin error | user, 2026-08-14 |
| Horizon | Ends after R24 (no finals) | user, 2026-08-14 |
| Double-points week | Exists, but **every tipster has already used theirs** | user, 2026-08-14 |
| Margin tips remaining | 2 (first game of each round: R23G1, R24G1) | brief §1 |

The joker answer is load-bearing: because no double-points week remains, every
remaining game is worth exactly 1 point, so the "disagreeing rivals all move
together by ±1" invariant holds and the state space stays tractable.

Consequence: DeanLFC's 145 is **points**, not tips (6 correct × 2 in R22). The
field is named `points` throughout so nobody re-derives a wrong tip count from it.
A per-tipster `joker_available` flag defaults to `False`; if one turns out to be
unspent it is a config change, not a redesign.

## Standings after R22

| Tipster | Points | Margin error | Gap `g_i` = `points_i − points_me` |
|---|---|---|---|
| Ryan Board | 147 | 628 | **+1** |
| **Me** | 146 | **573** | — |
| NRL > AFL | 146 | 677 | 0 |
| Markash | 145 | 574 | −1 |
| DeanLFC | 145 | 578 | −1 |
| Mikefooty | 145 | 587 | −1 |
| ADRIANOartini | 145 | 646 | −1 |

I hold the lowest margin error of anyone in contention, so a tie is a win for me
against every current rival.

## Core structure

Track **differentials**, not scores. Let `delta_j` be tipster `j`'s net points
versus an all-favourites baseline. Every tipster's final score is
`points_j + F + delta_j`, where `F` is the number of remaining games won by the
favourite — the *same* `F` for everyone. `F` cancels:

```
final differential against rival j  =  g_j + delta_j − delta_me
```

`delta` moves by `+1` when you tip the dog and it wins, `−1` when you tip the dog
and it loses, and `0` whenever you tip the favourite.

## Opponent model

Per user instruction: **the leader tips favourites; the chasers do not.** Chasers
at 145 need +2 on Ryan, so they will take dogs, and a lucky chaser overtakes me
even in worlds where I beat Ryan. Ignoring that overstates my win probability.

Each rival is modelled as playing their own level-0 best response: they maximise
their own `P(first)` assuming *everyone else* tips favourites. This makes each
rival's deviation schedule a deterministic function of their own state, and
critically **independent of my choices** — so their deltas can be carried as exact
extra DP dimensions rather than approximated.

Rivals sharing an identical policy share a delta and collapse to one dimension.
Grouping is computed at runtime, not hardcoded.

### Rival terminal conditions (level-0 view)

Rival `j` wins if `points_j + delta_j` exceeds every other tipster's, with ties
resolved on countback. Note ADRIANOartini's asymmetry: his 646 margin error
**loses** the countback to Ryan's 628, so he needs +3 outright where Markash (574),
DeanLFC (578) and Mikefooty (587) need only +2. This is a genuine strategic
asymmetry and it is in my favour.

## My DP

State `(t, delta_me, delta_group0, delta_group1, ...)`, `t` indexing games in lock
order. Ryan contributes no dimension (`delta ≡ 0`).

```
V(T, ·)   = terminal(·)
V(t, ·)   = max over my action a in {favourite, dog} of
              p_t · V(t+1, ·| favourite wins) + (1−p_t) · V(t+1, ·| dog wins)
```

Rival actions at each node come from their precomputed policy tables. Top-down
memoised recursion visits reachable states only. Elimination pruning: if
`1 − delta_me` exceeds the number of games remaining, Ryan is uncatchable → value 0.

### Terminal value

```
diff_j = g_j + delta_group(j) − delta_me
if any diff_j > 0:  0.0                    (someone beat me outright)
else:               P(I win the countback against every j with diff_j == 0)
```

Rivals with `diff_j < 0` contribute nothing regardless of how far below they are, so
the terminal depends only on **which subset is tied at zero** — 2⁶ = 64 cases,
precomputed once into a lookup table.

## Margin countback model

Derived from the line rather than from tipster history (user's choice).

Two margin games remain. For each, actual margin `M ~ Normal(line, sigma)` with
`sigma = 37`; tipster `j`'s margin tip `~ Normal(line, tau)`; error contribution
`|M − tip|`. Cumulative error is the current value plus both games' contributions.

Two structural facts, both in my favour, that the brief's random-walk framing missed:

1. **The gap movement is bounded.** By the reverse triangle inequality,
   `| |M−m_i| − |M−m_j| | ≤ |m_i − m_j|` — two tipsters' error gap can move by at
   most the difference between their two margin tips, whatever the actual margin
   does. The walk is not unbounded, so my 55-point lead over Ryan is more durable
   than `sigma_diff = 22/round` implies.
2. **Tipping the line minimises expected error**, because the median of `M`
   minimises `E|M − m|`. This derives, rather than assumes, the "tip the line,
   protect the asset" prior.

`tau` defaults to 10, implied by R22's 28-point error spread across 7 tipsters, and
is swept in the sensitivity output. It is the model's main unvalidated assumption
and is labelled as such in every output.

**Correlation is handled exactly.** Every tipster's error is computed against the
same actual margin `M`, so countback outcomes are *not* independent across rivals;
multiplying pairwise `q_i` together would overstate my safety. The 64-entry table is
built by simulating `M` directly, which captures the joint distribution. (Rival-vs-
rival terminals use the pairwise product — an approximation, but it affects only how
rivals are modelled to behave, not my own win probability.)

## Scope

**In:** CSV input, three devig methods, rival policy solve, joint DP, next-game
recommendation with reasoning, margin tip recommendation and sensitivity, baseline
comparison, devig sensitivity. Pure stdlib, Python 3.9-compatible.

**Out, deliberately:** the full multiset DP with lock-vs-result timing states;
equilibrium / fictitious play; disk caching; XLSX output; numba.

**What that costs:** rivals are modelled as level-0 players, so the engine does not
capture how Ryan would *respond* to my deviation. That matters at R24G1 and
essentially nowhere before it.

## Testing

- Devig closed forms: odds 1.02/15.00 give 6.4% proportional, 3.6% odds-ratio.
  Verified before build.
- Differential transition: tip favourite → delta unchanged; tip dog → ±1.
- Terminal value: `delta_me = 0` loses to Ryan; `delta_me ≥ 2` beats him outright;
  `delta_me = 1` ties him and resolves on countback.
- Countback: tipping the line beats tipping off the line in expected error.
- Monotonicity: `P(win)` non-decreasing in `delta_me`.

## Known limitations

1. The §5 acceptance values (51.25% / 21.10%) cannot be verified — `quick_solver.py`
   and `AFL_Tipping_Inputs.xlsx` were not present on this machine, and the real R23/R24
   odds are not yet loaded. The check runs once real odds are pasted in.
2. `tau` is assumed, not fitted. Filling past rounds' margin tips would measure it.
3. Rivals are level-0; they do not respond to my play.
