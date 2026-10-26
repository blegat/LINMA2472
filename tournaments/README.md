# Reversi AI tournament

[Open the tournament viewer](https://blegat.github.io/LINMA2472/tournaments/).

Explore the standings, head-to-head records and games played by the student AIs.
Select an available tournament snapshot, filter games by stage or player, or
select a cell in the head-to-head table to see that matchup's games.

## Reading the results

- **Final standings** show the tournament's final ranking after the ladder stage.
- **Wins, draws and losses** include all recorded games, across both the round
  robin and ladder. **Disc ±** is the total number of discs scored minus conceded.
- **Head to head** shows the row player's wins, draws and losses against each
  opponent.
- **Qualification details** explains each status in the selected snapshot and
  lists every group's outcome. The Notes column gives any specific reason for
  exclusion or a failed eligibility check.

Pure deep-learning players can qualify.
Tree or graph search, opening books and handcrafted move-selection heuristics
are excluded, including when combined with a neural network. Players must also
pass the loading and legal-play checks within the move time limit. Older snapshots
retain the qualification rules and outcomes recorded at that time.

Use the links at the bottom of the viewer to download the raw results and games.
If a snapshot has no recorded games, the standings show the round-robin wins and
losses from its results file instead.

## Run the viewer locally

From the root of this public repository, using Python 3.9 or later:

```sh
python3 tournaments/generate_index.py
python3 -m http.server 8000
```

Open <http://localhost:8000/tournaments/> in a browser with JavaScript enabled.
No third-party Python packages are required.
