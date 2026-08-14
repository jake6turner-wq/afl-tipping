# Named input sets — current vs scenario — design

Date: 2026-08-14

## Objective

Let one engine run against two independent worlds: **reality**, which mirrors the
comp site, and a **sandbox**, where hypothetical results and odds can be edited
freely without touching the real position. Switching between them must be one flag,
and it must be impossible to confuse a sandbox recommendation for a real one.

## Motivation

`--fixtures` and `--leaderboard` already accept arbitrary paths, so a second world
is technically reachable today. Three things are missing:

1. **Pairing.** The leaderboard and the fixtures must move together. Two independent
   path flags let you run reality's leaderboard against the sandbox's fixtures, which
   is silently wrong rather than an error.
2. **Ergonomics.** Two long paths per invocation, retyped every run.
3. **Output separation.** Both worlds write `output/recommendation.csv`. A sandbox run
   overwrites the real one, and the file carries nothing identifying which world
   produced it.

Point 3 is the load-bearing one. The engine's whole output is a single recommended
tip; a stale or mislabelled one is worse than none.

## Concept

A **set** is a named directory holding exactly one `leaderboard.csv` and one
`fixtures.csv`. Sets are resolved by name at runtime, never hardcoded.

```
inputs/current/{leaderboard,fixtures}.csv     reality — mirrors the comp site
inputs/scenario/{leaderboard,fixtures}.csv    sandbox — hypotheticals
output/current/recommendation.csv
output/scenario/recommendation.csv
```

Set names are arbitrary, so the design is not limited to two: `--set whatif-r23`
resolves `inputs/whatif-r23/` and `output/whatif-r23/` with no code change.

## Resolution

One helper is the single source of truth for path construction:

```python
def resolve_set(name: str) -> SetPaths:
    """inputs/<name>/{leaderboard,fixtures}.csv and output/<name>/."""
```

`LEADERBOARD_CSV` and `FIXTURES_CSV` are repointed at `inputs/current/`. They remain
module-level constants serving as the argparse defaults and the `load_leaderboard` /
`load_fixtures` parameter defaults, so both loaders keep their current signatures and
every existing caller — including the tests that pass explicit paths — is unaffected.

**Precedence:** an explicit `--fixtures` or `--leaderboard` overrides the set-derived
path for that file only. This preserves the existing one-off escape hatch. It also
re-opens the mismatch hazard from motivation 1, so overriding exactly one of the two
prints a warning naming both resolved paths.

## CLI surface

| Command | Effect |
|---|---|
| `--recommend` | Runs `--set current` (the default) |
| `--recommend --set scenario` | Same engine, sandbox inputs, sandbox output |
| `--copy-set scenario` | Clones `inputs/current/` into `inputs/scenario/` |
| `--copy-set scenario --force` | Same, skipping the confirmation prompt |
| `--make-template --set scenario` | Writes a blank set into `inputs/scenario/` |

`--copy-set` always reads from `current/`. Forking one scenario from another is
deliberately not supported: reality stays the only source of truth, which keeps the
mental model to one hop. Generalising to `--copy-set SRC DST` is a later change if the
need appears.

`--copy-set` fails if `inputs/current/` is missing, and prompts before overwriting an
existing target. The prompt is the only interactive element in the tool; `--force`
exists so the behaviour stays scriptable.

A missing or malformed set raises the existing `InputError` type, naming the directory
searched and suggesting `--make-template --set <name>`, matching the established error
style of `load_fixtures`.

## Guarding against misreading a scenario as reality

The one genuine hazard: acting on a hypothetical because the set was forgotten. Three
independent defences, none of which rely on remembering anything:

- The report banner prints the active set name and both resolved input paths.
- Any set other than `current` prints `*** SCENARIO SET -- NOT REALITY ***` in the
  header, alongside the existing `ASSUMPTIONS` block.
- `recommendation.csv` gains a `set` row, so the artefact is self-identifying once
  detached from the terminal that produced it.

## Migration

`git mv` the two existing CSVs from `inputs/` into `inputs/current/`, then seed
`inputs/scenario/` from them with the new `--copy-set scenario`. The scenario set
therefore starts as an exact copy of the R23/R24 fixture and the post-R22 leaderboard.

`inputs/fixtures_PLACEHOLDER.csv` stays at `inputs/` root. It is reference material
for the CSV format, not an input set, and giving it a set directory would imply it is
runnable.

## Testing

New cases in `test_tipping.py`, extending `TestInputValidation`:

- `resolve_set` builds the expected input and output paths for a given name.
- An explicit `--fixtures` path overrides the set-derived fixtures path.
- Overriding exactly one of the two paths warns.
- A missing set directory raises `InputError` naming the directory and `--make-template`.
- `--copy-set` refuses when the source set is absent.
- `--copy-set` does not overwrite an existing target without confirmation or `--force`.

Set resolution and the copy helper are pure path logic, testable without invoking the
solver, so these add negligible runtime to the existing suite.

## Documentation

README gains a short section covering the two sets, the `--set` flag, the `--copy-set`
refresh workflow, and the output split. The existing "Filling in `inputs/fixtures.csv`"
section is repathed to `inputs/current/fixtures.csv`.

## Explicitly out of scope

- Applying real results to a set automatically. Updating points and margin errors after
  each round stays a manual edit of `leaderboard.csv`.
- Diffing two sets, or reporting one against the other.
- Any change to the solver, the devig, the rival model, or the countback. This design
  touches path resolution, the CLI, and reporting labels only.
