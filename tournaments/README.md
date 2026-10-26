# Reversi tournament viewer

The viewer uses the original `results.csv` and `games.csv` produced by the private
Julia tournament runner. All standings displays, game summaries and head-to-head
records are calculated in the browser. A single viewer serves every snapshot, so
changing the visualization also changes how historical raw data is displayed.

## Publish another snapshot

Copy the unmodified files into a dated folder in this public repository:

```text
tournaments/
  2026-10-26/
    results.csv
    games.csv
  2026-11-02/
    results.csv
    games.csv
```

Commit and push the new CSVs through the normal repository workflow. No manifest,
HTML, derived statistics or visualization output needs to be maintained by hand.
The existing GitHub Pages workflow runs `generate_index.py`, which discovers the
dated folders and writes the small `runs.json` file used by the snapshot selector.
Folders must be named `YYYY-MM-DD`. A run with only `results.csv` is supported,
including historical runs with a single qualified player and no recorded games.

The viewer is available at `tournaments/` on the course website. A link such as
`tournaments/?run=2026-10-26` selects a particular snapshot.

## Local preview and checks

From the repository root:

```sh
python3 tournaments/generate_index.py
python3 -m http.server 8000
```

Open `http://localhost:8000/tournaments/`. To run the focused checks (Node 18+ and
Python 3.9+, no third-party packages):

```sh
node --test tournaments/test/*.test.js
python3 -m unittest discover -s tournaments/test -p 'test_*.py'
```

Final ranks come directly from `results.csv`; the viewer does not rerank players.
Win/draw/loss records and disc differences include every row of `games.csv`, across
both the round robin and ladder. When no games file exists, round-robin wins and
losses from the results file are shown instead. Historical qualification statuses
are retained as recorded, rather than reinterpreted using today's eligibility rules.
