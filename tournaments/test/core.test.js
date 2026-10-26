import test from "node:test";
import assert from "node:assert/strict";
import { parseCSV, readResults, readGames, standings, matchup, statusName, tournamentPoints, totalPoints } from "../core.js";

const resultsCSV = 'group,name,commit,status,explanation,phase1_wins,phase1_losses,phase1_ranking,final_ranking\n' +
  '4,Megatron,abc123,allowed,"",2,0,1,1\n2,"",def456,allowed,"",0,2,2,2\n1,"",xyz789,positional_ai,"",,,,\n';
const gamesCSV = 'phase,round,black,white,winner,black_discs,white_discs,moves,forfeit\n' +
  '1,1,2,4,4,27,37,60,""\n1,1,4,2,4,44,20,60,""\n' +
  '2,1,4,2,4,44,20,60,""\n2,1,2,4,4,27,37,60,""\n';

test("CSV preserves commas, escaped quotes, Unicode, multiline notes and empty fields", () => {
  assert.deepEqual(parseCSV('\uFEFFgroup,name,notes,rank\r\n2,"Un ""modèle""","line one\nline two, ✓",\r\n'), [
    { group: "2", name: 'Un "modèle"', notes: "line one\nline two, ✓", rank: "" },
  ]);
  assert.deepEqual(parseCSV(""), []);
  assert.deepEqual(parseCSV("group,status\n"), []);
  assert.throws(() => parseCSV('group,name\n2,"unfinished'), /Unclosed/);
  assert.throws(() => parseCSV("group,name\n2"), /number of fields/);
  assert.throws(() => parseCSV("group,group\n2,2"), /unique/);
});

test("raw results retain authoritative ranks while records include both stages", () => {
  const players = readResults(resultsCSV);
  const games = readGames(gamesCSV);
  const ranks = standings(players, games);
  assert.equal(players.length, 3);
  assert.equal(games.length, 4);
  assert.deepEqual(ranks.map(row => [row.id, row.rank, row.wins, row.draws, row.losses, row.difference]),
    [["4", 1, 4, 0, 0, 68], ["2", 2, 0, 0, 4, -68]]);
  assert.equal(ranks[0].initialWins, 2);
  assert.deepEqual(matchup(games, "4", "2"), { played: 4, wins: 4, draws: 0, losses: 0 });
  assert.deepEqual(matchup(games, "2", "4"), { played: 4, wins: 0, draws: 0, losses: 4 });
});

test("draws, forfeits and colour changes have consistent player-relative records", () => {
  const players = [
    { id: "1", rank: 2, status: "allowed" },
    { id: "2", rank: 1, status: "allowed" },
  ];
  const games = readGames('phase,round,black,white,winner,black_discs,white_discs,moves,forfeit\n' +
    '1,1,1,2,draw,32,32,60,""\n2,1,2,1,2,3,1,0,"group 1: illegal move"\n');
  assert.deepEqual(standings(players, games).map(row => [row.id, row.wins, row.draws, row.losses, row.difference]),
    [["2", 1, 1, 0, 2], ["1", 0, 1, 1, -2]]);
  assert.deepEqual(matchup(games, "1", "2"), { played: 2, wins: 0, draws: 1, losses: 1 });
  assert.throws(() => readGames("phase,black,white,winner\n1,1,2,3"), /invalid/);
});

test("historical snapshots work without games or the current eligibility policy", () => {
  const text = 'group,name,status,phase1_wins,phase1_losses,phase1_ranking,final_ranking\n' +
    '4,Osmia,allowed,0,0,1,1\n2,,non-transformer,,,,\n';
  const players = readResults(text);
  const ranks = standings(players, []);
  assert.equal(ranks.length, 1);
  assert.equal(ranks[0].id, "4");
  assert.equal(ranks[0].wins, 0);
  assert.equal(statusName(players.find(player => player.id === "2").status), "Excluded by policy (legacy)");
  assert.equal(standings(players.map(player => ({ ...player, status: "failed" })), []).length, 0);
});

test("points use qualified final positions and halve at each place", () => {
  for (const [rank, expected] of [[1, 0.2], [2, 0.1], [3, 0.05], [4, 0.025], [5, 0.0125]]) {
    assert.equal(tournamentPoints({ status: "allowed", rank }), expected);
  }
  for (const status of ["positional_ai", "failed", "non-dl", "non-transformer", "unknown"]) {
    assert.equal(tournamentPoints({ status, rank: 1 }), 0);
  }
  for (const rank of [null, undefined, 0, -1, 1.5]) {
    assert.equal(tournamentPoints({ status: "allowed", rank }), 0);
  }
  assert.ok(tournamentPoints({ status: "allowed", rank: 40 }) > 0);
  const players = readResults(resultsCSV);
  players[0].initialRank = 2;
  assert.equal(standings(players, readGames(gamesCSV))[0].points, 0.2);
});

test("totals sum by group across dates, keep the latest name and include zero-point groups", () => {
  const snapshots = [
    { id: "2026-11-02", players: [
      { id: "4", name: "Megatron", status: "allowed", rank: 2 },
      { id: "2", name: "Second name", status: "failed", rank: 1 },
      { id: "3", name: "New player", status: "allowed", rank: 1 },
    ] },
    { id: "2026-10-26", players: [
      { id: "4", name: "Osmia", status: "allowed", rank: 1 },
      { id: "2", name: "First name", status: "allowed", rank: 2 },
      { id: "1", name: "", status: "positional_ai", rank: null },
    ] },
  ];
  const before = structuredClone(snapshots);
  const totals = totalPoints(snapshots);
  assert.deepEqual(totals.map(group => group.id), ["4", "3", "2", "1"]);
  assert.ok(Math.abs(totals[0].total - 0.3) < Number.EPSILON);
  assert.equal(totals[0].name, "Megatron");
  assert.equal(totals[0].qualifiedRuns, 2);
  assert.equal(totals[1].total, 0.2);
  assert.equal(totals[2].total, 0.1);
  assert.equal(totals[2].name, "Second name");
  assert.deepEqual(totals[3], { id: "1", name: "", total: 0, qualifiedRuns: 0, rank: null });
  assert.deepEqual(snapshots, before);
  assert.deepEqual(totalPoints([]), []);
  assert.throws(() => totalPoints([snapshots[0], snapshots[0]]), /counted twice/);
});

test("mathematically equal cumulative scores share a rank", () => {
  const snapshots = [1, 2, 3].map((number, index) => ({
    id: `2026-11-0${number}`,
    players: [
      { id: "1", status: "allowed", rank: index === 0 ? 1 : 3 },
      { id: "2", status: "allowed", rank: 2 },
    ],
  }));
  assert.deepEqual(totalPoints(snapshots).map(group => group.rank), [1, 1]);
});
