# AFL tipping-comp optimal strategy engine

Answers one question: **what do I tip in the next game that locks, and what is my
probability of winning the comp under each option?**

Not a tip predictor. Everyone in the comp has the same information and will tip much
the same way. The edge is entirely in *when to differentiate*.

Pure Python standard library — no numpy, pandas or openpyxl. Runs on the system
Python 3.9.

## Use

```sh
python3 tipping.py --make-template            # write inputs/current/*.csv
# ... fill in inputs/current/fixtures.csv ...
python3 tipping.py --recommend                # the next decision
python3 tipping.py --recommend --explain      # + contingency table for 3 games
python3 test_tipping.py                       # 118 tests, ~13s
```

## Input sets

Inputs live in named **sets**. A set is a directory pairing one leaderboard with
one fixture list, so the two can never be mixed across sets by accident.

```
inputs/current/     reality -- keep this matching the comp site
inputs/scenario/    sandbox -- edit freely to test what-ifs
output/current/     output/scenario/
```

```sh
python3 tipping.py --recommend                  # reads inputs/current/
python3 tipping.py --recommend --set scenario   # reads inputs/scenario/
python3 tipping.py --copy-set scenario          # reset the sandbox from current/
```

Set names are arbitrary, so `--set whatif-r23` works with no code change. Any set
other than `current` prints a `*** SCENARIO SET -- NOT REALITY ***` banner and
writes to its own output directory, so a sandbox run can never overwrite a real
recommendation or be mistaken for one afterwards.

`--fixtures` and `--leaderboard` still take explicit paths and override `--set`
for that one file. Overriding exactly one of the two prints a warning, because it
pairs a leaderboard from one world with fixtures from another.

Useful flags: `--devig {proportional,odds_ratio,shin}` (default `odds_ratio`),
`--tau` (rival margin-tip dispersion), `--reluctance` (rivals' aversion to heavy
underdogs), `--sims`, `--seed`, `--set`, `--fixtures`, `--leaderboard`.

## Filling in `inputs/current/fixtures.csv`

One row per **remaining** game, in lock order. Delete rows once played.

| column | meaning |
|---|---|
| `game_id` | e.g. `R23G1` |
| `lock_local` | free text, printed back to you |
| `home` / `away` | team names |
| `home_odds` / `away_odds` | decimal head-to-head odds |
| `line_fav` | favourite's expected winning margin, **positive** |
| `is_margin_game` | `1` for the first game of each round, else `0` |
| `p_home_override` | optional; your own model's P(home win). Beats the devig. |

`inputs/current/leaderboard.csv` needs exactly one row with `is_me=1`.

> `points`, not tips — DeanLFC's 145 includes a doubled round. Every remaining game
> is worth 1 point because all double-point weeks are spent.

## How it works

Scores are tracked as **differentials**. Every tipster's final score is
`points + F + delta`, where `F` is however many favourites win — the same `F` for
everyone, so it cancels. `delta` is your net points versus tipping every favourite:
`+1` per dog you took that won, `−1` per dog that lost. Tipping the favourite
alongside everyone else cannot change your position, which is why *always favourite*
scores exactly 0%.

Rivals are modelled as level-0 players maximising their own `P(first)`. Whoever
currently leads on points tips favourites; chasers deviate to close the gap. The
leader is read from the board, so editing a scenario's leaderboard reassigns the
role. Rivals sharing a policy share a DP dimension. My policy is exact backward
induction over `(game, my delta, each rival group's delta)`.

Chasers are not pure expected-value maximisers: `--reluctance` (default `0.10`) is
how much extra win probability one needs before backing a heavy underdog, scaled by
price as `reluctance x (p_fav - 0.5)`. At `0` a chaser takes a 15.00 outsider as
readily as a coin flip; at `0.10` they deviate in the 58-76% games and leave the
85%+ games alone. It applies to rivals only, so your own recommendation stays an
exact optimum against them.

A tie for first goes to the lowest cumulative margin error, so the terminal value is
`P(I win the countback against everyone tied with me)` — a 64-entry table built by
simulating the actual margin, which captures the fact that everyone's error is
measured against the *same* result and so is correlated across rivals.

The next tip is chosen **by the simulation itself**. Every season is drawn once and
played twice — once tipping the favourite, once the underdog — over identical results
and identical margin draws, and the branch with the higher eventual win rate wins. So
the headline `P(win comp)` and your row in `WHO WINS THE COMP` are the same number
from the same run.

From the second game onward you play the same level-0 rule as everyone else, which
makes this exact one-step lookahead over a rollout policy rather than a full optimum.
The exact grouped solve still runs and is printed as a cross-check; a `WARNING` fires
if the two models pick different tips, or if the edge is inside the sampling error.

`WHO WINS THE COMP` comes from the same **season simulation**
(`--sim-seasons`, default 60000). Each season is played out game by game: everyone
re-decides from the live standings after every result, one result is drawn and shared
by all of them, and whoever leads *at that moment* tips the favourite — so the
always-favourite role passes around as the lead changes. Nobody shares a policy, so a
rival who decides to chase a run of upsets is making their own choice.

`YOUR NEXT DEVIATION` answers "when do I next tip an underdog, and who". It is a
**distribution over games**, not a plan: whether you deviate at a given game depends
on how the results fall before it, so there is no fixed sequence of future tips —
only a rule that answers the standings. Read the likeliest row as "the next underdog
you are most likely to need", and the `never deviate` row as the chance you can ride
favourites home.

`CONTINGENCY TABLE` (`--explain`) comes from the same simulation, so its `delta +0`
row for the next game **is** the headline recommendation rather than a second opinion
on it. Cells whose two branches sit inside the sampling error are marked
`too close to call` instead of being given a spurious winner. It costs roughly 15
extra simulations, so it runs at `--contingency-seasons` (default 6000) rather than
the headline's 60000, and `--explain` takes about a minute.

The recommendation is broken down by **what actually happens in the next game**, so
you can see how much this one result decides. Tipping with the field is low variance
(everyone moves together, so the result barely changes your position); taking the dog
puts the season on one game. On the live board those two swings were 0.2% and 58.2%.

### `--equilibrium`

By default every tipster plays a **level-0** rule: they pick their best action
assuming the rest of the field tips favourites. Their own deviations are therefore
invisible to each other, and your recommendation is one-step lookahead over that
rollout.

`--equilibrium` replaces it with a policy learned by **backward induction over
simulated seasons**. Working from the last game to the first, it samples seasons that
reach each game, has each tipster try both actions there while the others answer with
the current rule, and finishes on the policy already settled for every later game.
Because the last game is decided first, each action is judged against later games
already being played well rather than against a heuristic. Repeated sweeps at a fixed
game are iterated best response within that stage.

Three things keep the sampling affordable:

- **Both arms share one drawn season.** A position can no longer be reached with one
  action sampled and the other missing, which used to throw the position away.
- **Every tipster is scored from that season**, not one randomly chosen focus, so a
  single rollout yields `2 x field size` samples instead of one.
- **Games priced under $1.15 are conceded.** If the whole field tips one side, every
  score moves together and no gap changes, so the game is a differential no-op and
  skipping it is exact given the assumption. The report names the games it skipped.

```sh
python3 tipping.py --recommend --equilibrium
python3 tipping.py --recommend --equilibrium --equilibrium-seasons 10000
```

`--equilibrium-seasons` is the dial worth turning; sweeps buy far less. Measured on
the live set, quadrupling seasons took learned states from 149 to 257 while tripling
sweeps moved it 149 to 161 for triple the runtime. Coverage, not iteration, is the
binding constraint.

`--equilibrium-gap-clamp N` buckets point gaps wider than `N` into `N` when keying the
learned table, so near-identical positions share an entry. It is **lossy** — being one
clear is not being six clear — so it is off by default. It buys a lot of coverage
(28.8% of lookups answered to 73.4% at `N=1` on the live set) but the states it pools
disagree with each other, so the flip rate goes *up*: it is a coarser map, not a
better-learned one. The fallback always receives the true gaps regardless.

The block also prints **what every tipster plays in the upcoming game** under the
learned rule, with each one's gap to the lead, so you can see the shape of the round
rather than just your own tip.

It is **approximate, not a proven equilibrium**, and it says so: the report prints how
many states it learned, the thinnest sample behind any of them, how often it fell back
to level-0 on an unseen state, and **what share of states still changed action on the
final sweep**. That last figure is a rate rather than a boolean, because across
hundreds of sampled states noise alone flips a few every run — a "nothing moved" flag
could only ever read False. Under 2% counts as converged.

It is not there yet. On the live set it answers about 29% of lookups with 26% of
states still moving, so treat the section as a cross-check on the headline, not as an
independent answer. Exact equilibrium is not available at any budget — ten players
over nine games is ~10^10 joint states with 1024 action profiles per node.

Every tipster shares one learned rule keyed by their own situation. That is what makes
it tractable, and it cannot express an idiosyncratically reckless rival.

Everything below the table — baselines, chaser sensitivity, devig sensitivity, the
margin-tip sweep — still comes from the exact grouped solve, which is a different
model with different absolute numbers. Read those as **relative comparisons**, not
against the headline.

## Read the output critically

- **`tau = 10` is assumed, not fitted.** It is the dispersion of rivals' margin tips
  around the line, inferred from a single round's 28-point error spread. Filling in
  past rounds' margin tips would measure it.
- **`reluctance = 0.10` is assumed, not fitted.** It was calibrated to separate the
  85%+ games from the 58-76% band on this fixture, not measured from anyone's actual
  tipping. Sweep it with `--reluctance` and check the recommendation holds.
- **The chaser model moves the answer by ~7 points.** A pure level-0 chaser believes
  the field is frozen and stops deviating once they're far enough ahead, which
  flatters you. `CHASER MODEL SENSITIVITY` reports the relentless variants; check
  the recommended action is stable across them.
- **The tip recommendation still groups rivals.** `solve_joint` must: ungrouped, its
  state is seven independent deltas, around `41^7` nodes against the ~26 the grouped
  version reaches. Only the win-probability table is ungrouped.
- **Margin error drives how hard a rival chases.** A tipster who would lose a tie has
  to win outright, so they deviate more and carry more variance. That is why
  ADRIANOartini (647) consistently finishes above Mikefooty (596) despite the worse
  error and the same points: needing more forces him to try more.
- **Rivals don't respond to you.** No equilibrium solve, so the engine can't tell you
  how the leader would counter your deviation. That matters at the last game or two
  and essentially nowhere before.
- **Watch the two `WARNING` lines.** They fire when the recommended action flips
  between devig methods or chaser models, which means the decision is genuinely
  marginal rather than merely close.

## Layout

```
tipping.py         the engine (devig, countback, rival model, joint DP, report)
test_tipping.py    tests
inputs/current/    leaderboard.csv, fixtures.csv -- reality
inputs/scenario/   leaderboard.csv, fixtures.csv -- sandbox
output/<set>/      recommendation.csv
docs/superpowers/  specs/ and plans/
```
