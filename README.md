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
python3 test_tipping.py                       # 78 tests, ~2.4s
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

`--recommend` also prints `WHO WINS THE COMP`: every tipster's probability of
finishing first in one shared world, where each plays their own best response. It is
a real distribution and sums to 100%, and your row equals the headline exactly.

Read the marked rows carefully. Rivals with identical policies are collapsed onto one
delta dimension, so the engine has them tipping alike for the rest of the season and
their order frozen at today's points — anyone sharing a group with a higher-pointed
rival shows 0% by construction, not by form.

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
- **Rivals sharing a policy are perfectly correlated.** Collapsing them onto one
  delta assumes they tip identically all season, which freezes their relative order.
  It is harmless for your own number — what matters is whether *any* of them beats
  you — but it makes the per-person split inside a group meaningless.
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
