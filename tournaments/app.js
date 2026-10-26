import { readResults, readGames, playerName, stageName, statusName, standings, matchup } from "./core.js";

const $ = id => document.getElementById(id);
const state = { players: [], games: [], ranked: [], pair: null, sort: "rank", direction: 1 };
let controller;
const dateLabel = id => new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "long", year: "numeric" })
  .format(new Date(`${id}T12:00:00`));
const findPlayer = id => state.players.find(player => player.id === id) || { id };
const signed = number => number > 0 ? `+${number}` : String(number);

function node(tag, text = null, className = "") {
  const element = document.createElement(tag);
  if (text !== null) element.textContent = text;
  if (className) element.className = className;
  return element;
}
function cell(text, className = "") { return node("td", text, className); }
function playerCell(player, colour) {
  const wrapper = node("div", null, colour ? "colour-player" : "");
  if (colour) wrapper.append(node("span", null, `disc ${colour}`));
  const label = node("div");
  label.title = playerName(player);
  label.append(node("span", playerName(player), "player-name"));
  if (player.name) label.append(node("span", `Group ${player.id}`, "player-group"));
  wrapper.append(label);
  return wrapper;
}
function recordBar(player) {
  const bar = node("div", null, "record-bar");
  bar.setAttribute("role", "img");
  bar.setAttribute("aria-label", `${player.wins} wins, ${player.draws} draws, ${player.losses} losses`);
  bar.title = bar.getAttribute("aria-label");
  const total = player.wins + player.draws + player.losses;
  for (const [type, count] of [["win", player.wins], ["draw", player.draws], ["loss", player.losses]]) {
    const segment = node("span", null, `record-${type}`);
    segment.style.width = `${total ? count / total * 100 : 0}%`;
    bar.append(segment);
  }
  return bar;
}

function renderStandings() {
  const ranked = [...state.ranked].sort((a, b) => {
    const first = state.sort === "name" ? playerName(a) : a[state.sort];
    const second = state.sort === "name" ? playerName(b) : b[state.sort];
    const comparison = typeof first === "string" ? first.localeCompare(second) :
      (first ?? Infinity) - (second ?? Infinity);
    return state.direction * comparison || Number(a.id) - Number(b.id);
  });
  $("standings-body").replaceChildren(...ranked.map(player => {
    const row = node("tr", null, player.rank === 1 ? "leader" : "");
    const rank = cell(null); rank.append(node("span", player.rank ?? "—", "rank-chip"));
    const name = cell(null); name.append(playerCell(player));
    const record = cell(null, "record-column"); record.append(recordBar(player));
    const change = player.initialRank !== null && player.rank !== null ? player.initialRank - player.rank : 0;
    const delta = cell(change ? `${change > 0 ? "↑" : "↓"} ${Math.abs(change)}` : "—", `number ${change > 0 ? "positive" : change < 0 ? "negative" : "muted"}`);
    delta.title = "Change from round-robin rank to final rank";
    row.append(rank, name, cell(player.wins, "number"), cell(player.draws, "number muted"),
      cell(player.losses, "number"), record,
      cell(signed(player.difference), `number ${player.difference > 0 ? "positive" : player.difference < 0 ? "negative" : "muted"}`), delta);
    return row;
  }));
  $("no-players").hidden = state.ranked.length > 0;
  document.querySelectorAll("[data-sort]").forEach(button => {
    const heading = button.closest("th");
    heading.removeAttribute("aria-sort");
    if (button.dataset.sort === state.sort) heading.setAttribute("aria-sort", state.direction === 1 ? "ascending" : "descending");
  });
}

function renderMatrix() {
  const table = node("table", null, "matrix");
  table.style.minWidth = `${100 + state.ranked.length * 85}px`;
  const caption = node("caption", "Head-to-head records: each row player's wins, draws and losses.", "sr-only");
  const head = node("thead"), header = node("tr");
  header.append(node("th", "Player"));
  state.ranked.forEach(player => {
    const heading = node("th", `G${player.id}`); heading.title = playerName(player); heading.scope = "col";
    header.append(heading);
  });
  head.append(header);
  const body = node("tbody");
  state.ranked.forEach(player => {
    const row = node("tr"), label = node("th", playerName(player)); label.scope = "row"; row.append(label);
    state.ranked.forEach(opponent => {
      const value = cell(null);
      if (player.id === opponent.id) value.append(node("span", "·", "self"));
      else {
        const record = matchup(state.games, player.id, opponent.id);
        if (!record.played) value.append(node("span", "—", "empty-cell"));
        else {
          const button = node("button", `${record.wins} · ${record.draws} · ${record.losses}`,
            record.wins > record.losses ? "winning" : record.wins < record.losses ? "losing" : "tied");
          button.type = "button";
          button.title = `${playerName(player)} vs ${playerName(opponent)}: ${record.wins} wins, ${record.draws} draws, ${record.losses} losses`;
          button.setAttribute("aria-label", button.title);
          button.addEventListener("click", () => {
            state.pair = [player.id, opponent.id];
            $("player-filter").value = "all"; $("stage-filter").value = "all";
            renderGames();
            $("games-section").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
          });
          value.append(button);
        }
      }
      row.append(value);
    });
    body.append(row);
  });
  table.append(caption, head, body);
  $("matrix").replaceChildren(table);
  $("matrix").hidden = state.ranked.length === 0;
  $("no-matchups").hidden = state.games.length > 0;
}

function renderGames() {
  const phase = $("stage-filter").value, player = $("player-filter").value;
  const games = state.games.filter(game => (phase === "all" || game.phase === phase) &&
    (player === "all" || game.black === player || game.white === player) &&
    (!state.pair || (state.pair.includes(game.black) && state.pair.includes(game.white))));
  $("game-count").textContent = `${games.length} / ${state.games.length}`;
  $("match-filter").hidden = !state.pair;
  if (state.pair) $("match-filter-label").textContent = state.pair.map(id => playerName(findPlayer(id))).join(" vs ");
  $("games-body").replaceChildren(...games.map(game => {
    const row = node("tr"), stage = cell(null);
    stage.append(node("span", stageName(game.phase), `stage-tag ${game.phase === "2" ? "ladder" : ""}`), node("div", `Round ${game.round}`, "round-label"));
    const black = cell(null), white = cell(null);
    black.append(playerCell(findPlayer(game.black), "black")); white.append(playerCell(findPlayer(game.white), "white"));
    const scoreCell = cell(null), score = node("div", null, "score");
    const bar = node("span", null, "board-score"), segment = node("span");
    const total = game.blackDiscs + game.whiteDiscs;
    segment.style.width = `${total ? game.blackDiscs / total * 100 : 0}%`;
    bar.append(segment); bar.setAttribute("aria-hidden", "true");
    score.append(node("span", game.blackDiscs, "score-number"), bar, node("span", game.whiteDiscs, "score-number"));
    score.setAttribute("aria-label", `Black ${game.blackDiscs}, White ${game.whiteDiscs}`); scoreCell.append(score);
    const winner = cell(null);
    winner.append(node("span", game.winner === "draw" ? "Draw" : playerName(findPlayer(game.winner)), "winner-badge"));
    if (game.forfeit) {
      const details = node("details", null, "forfeit"); details.append(node("summary", "Forfeit"), node("p", game.forfeit)); winner.append(details);
    }
    row.append(cell(String(game.id).padStart(2, "0"), "muted"), stage, black, scoreCell, white, winner, cell(game.moves ?? "—", "number muted"));
    return row;
  }));
  $("no-games").hidden = games.length > 0;
  $("no-games").textContent = state.games.length ? "No games match these filters." :
    "No head-to-head games were recorded for this snapshot.";
}

function renderRoster() {
  const query = $("group-search").value.trim().toLocaleLowerCase();
  const players = [...state.players].sort((a, b) => Number(a.id) - Number(b.id)).filter(player =>
    `Group ${player.id} ${player.name} ${statusName(player.status)} ${player.explanation}`.toLocaleLowerCase().includes(query));
  $("roster-body").replaceChildren(...players.map(player => {
    const row = node("tr"), qualification = cell(null);
    qualification.append(node("span", statusName(player.status), `qualification ${player.status === "allowed" ? "qualified" : player.status === "failed" ? "failed" : ""}`));
    row.append(cell(`Group ${player.id}`), cell(player.name || "—"), qualification, cell(player.explanation || "—"));
    return row;
  }));
  $("no-groups").hidden = players.length > 0;
}

function renderRun(run) {
  state.ranked = standings(state.players, state.games); state.pair = null;
  state.sort = "rank"; state.direction = 1;
  $("stage-filter").value = "all"; $("group-search").value = "";
  $("player-filter").replaceChildren(new Option("All players", "all"),
    ...state.ranked.map(player => new Option(playerName(player), player.id)));
  const winner = state.ranked[0];
  $("winner-label").textContent = state.ranked.length === 1 ? "Top of the board · sole entrant" : "Top of the board";
  $("winner-name").textContent = winner ? playerName(winner) : "No qualified player";
  $("winner-rank").textContent = winner ? String(winner.rank ?? 1).padStart(2, "0") : "—";
  $("winner-detail").textContent = winner ? `Group ${winner.id} · ${winner.wins} wins · ${winner.draws} draws · ${winner.losses} losses` : "See qualification details for this snapshot.";
  $("stat-players").textContent = state.ranked.length;
  $("stat-games").textContent = state.games.length;
  $("stat-groups").textContent = state.players.length;
  $("roster-count").textContent = state.players.length;
  $("results-download").href = `${run.id}/results.csv`;
  $("games-download").href = `${run.id}/games.csv`; $("games-download").hidden = !run.games;
  $("snapshot-note").textContent = `${dateLabel(run.id)} · published snapshot`;
  renderStandings(); renderMatrix(); renderGames(); renderRoster();
  $("announcement").textContent = `${dateLabel(run.id)} loaded: ${state.ranked.length} qualified players and ${state.games.length} games.`;
}

async function getText(path, signal) {
  const response = await fetch(path, { signal, cache: "no-store" });
  if (!response.ok) throw new Error(`Could not load ${path} (HTTP ${response.status}).`);
  return response.text();
}
function showError(error) {
  $("loading").hidden = true; $("dashboard").hidden = true;
  $("error-message").textContent = `The results could not be loaded. ${error.message}`;
  $("error").hidden = false;
}
async function loadRun(run) {
  controller?.abort(); controller = new AbortController();
  const { signal } = controller;
  $("loading").hidden = false; $("error").hidden = true; $("dashboard").hidden = true;
  try {
    const [results, games] = await Promise.all([
      getText(`${run.id}/results.csv`, signal), run.games ? getText(`${run.id}/games.csv`, signal) : Promise.resolve(""),
    ]);
    if (signal.aborted) return;
    state.players = readResults(results); state.games = readGames(games);
    renderRun(run);
    const url = new URL(location.href); url.searchParams.set("run", run.id); history.replaceState(null, "", url);
    $("loading").hidden = true; $("dashboard").hidden = false;
  } catch (error) { if (error.name !== "AbortError") showError(error); }
}
async function boot() {
  $("error").hidden = true; $("empty").hidden = true; $("loading").hidden = false;
  try {
    const manifest = JSON.parse(await getText("runs.json"));
    if (!Array.isArray(manifest.runs) || manifest.runs.some(run => !/^\d{4}-\d{2}-\d{2}$/.test(run.id))) {
      throw new Error("The snapshot index is invalid.");
    }
    const runs = manifest.runs.sort((a, b) => b.id.localeCompare(a.id));
    if (!runs.length) { $("loading").hidden = true; $("empty").hidden = false; return; }
    const select = $("run-select");
    select.replaceChildren(...runs.map((run, i) => new Option(`${dateLabel(run.id)}${i === 0 ? " · latest" : ""}`, run.id)));
    const requested = new URL(location.href).searchParams.get("run");
    const chosen = runs.find(run => run.id === requested) || runs[0];
    select.value = chosen.id; select.disabled = false;
    select.onchange = () => loadRun(runs.find(run => run.id === select.value));
    await loadRun(chosen);
  } catch (error) { showError(error); }
}

document.querySelectorAll("[data-sort]").forEach(button => button.addEventListener("click", () => {
  const sort = button.dataset.sort;
  state.direction = state.sort === sort ? -state.direction : ["wins", "difference"].includes(sort) ? -1 : 1;
  state.sort = sort; renderStandings();
}));
$("stage-filter").addEventListener("change", renderGames);
$("player-filter").addEventListener("change", () => { state.pair = null; renderGames(); });
$("clear-match").addEventListener("click", () => { state.pair = null; renderGames(); });
$("group-search").addEventListener("input", renderRoster);
$("retry").addEventListener("click", boot);
boot();
