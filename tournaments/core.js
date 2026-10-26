// All display statistics are derived here from the original tournament CSVs.
export function parseCSV(source) {
  const rows = [];
  let row = [], field = "", quoted = false, closed = false;
  const pushField = () => { row.push(field); field = ""; closed = false; };
  const pushRow = () => { pushField(); if (row.some(value => value !== "")) rows.push(row); row = []; };
  const text = source.replace(/^\uFEFF/, "");
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (quoted) {
      if (char === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (char === '"') { quoted = false; closed = true; }
      else field += char;
    } else if (char === ",") pushField();
    else if (char === "\n" || char === "\r") {
      if (char === "\r" && text[i + 1] === "\n") i++;
      pushRow();
    } else if (closed && /[ \t]/.test(char)) continue;
    else if (closed) throw new Error("Unexpected text after a quoted CSV field.");
    else if (char === '"' && field === "") quoted = true;
    else if (char === '"') throw new Error("Unexpected quote in a CSV field.");
    else field += char;
  }
  if (quoted) throw new Error("Unclosed quoted CSV field.");
  if (field || row.length || closed) pushRow();
  if (!rows.length) return [];
  const headers = rows.shift().map(value => value.trim());
  if (headers.some(value => !value) || new Set(headers).size !== headers.length) {
    throw new Error("CSV column names must be present and unique.");
  }
  return rows.map((values, index) => {
    if (values.length !== headers.length) throw new Error(`CSV row ${index + 2} has an unexpected number of fields.`);
    return Object.fromEntries(headers.map((key, i) => [key, values[i]]));
  });
}

function integer(value, fallback = null) {
  if (value == null || value.trim() === "") return fallback;
  const result = Number(value);
  if (!Number.isSafeInteger(result) || result < 0) throw new Error(`Invalid numeric value: ${value}`);
  return result;
}

export function readResults(text) {
  const ids = new Set();
  return parseCSV(text).map(row => {
    if (!row.group?.trim() || !row.status?.trim() || ids.has(row.group)) {
      throw new Error("Each result needs a unique group and a qualification status.");
    }
    ids.add(row.group);
    return {
      id: row.group, name: (row.name || "").trim(), status: row.status,
      commit: row.commit || "", explanation: row.explanation || "",
      rank: integer(row.final_ranking), initialRank: integer(row.phase1_ranking),
      initialWins: integer(row.phase1_wins, 0), initialLosses: integer(row.phase1_losses, 0),
    };
  });
}

export function readGames(text) {
  return parseCSV(text).map((row, index) => {
    if (!row.black || !row.white || row.black === row.white || !row.phase ||
        ![row.black, row.white, "draw"].includes(row.winner)) {
      throw new Error(`Game ${index + 1} has invalid players, stage or winner.`);
    }
    return {
      id: index + 1, phase: row.phase, round: integer(row.round, 1),
      black: row.black, white: row.white, winner: row.winner,
      blackDiscs: integer(row.black_discs, 0), whiteDiscs: integer(row.white_discs, 0),
      moves: integer(row.moves), forfeit: row.forfeit || "",
    };
  });
}

export const playerName = player => player?.name || `Group ${player?.id ?? "?"}`;
export const stageName = phase => ({ "1": "Round robin", "2": "Ladder" }[phase] || `Stage ${phase}`);
export const statusName = status => ({
  allowed: "Qualified", positional_ai: "Default player", failed: "Failed eligibility",
  "non-dl": "Excluded by policy", "non-transformer": "Excluded by policy (legacy)",
}[status] || status);

export function standings(players, games) {
  return players.filter(player => player.status === "allowed").map(player => {
    const played = games.filter(game => game.black === player.id || game.white === player.id);
    const wins = played.filter(game => game.winner === player.id).length;
    const draws = played.filter(game => game.winner === "draw").length;
    const difference = played.reduce((sum, game) => sum + (game.black === player.id ? 1 : -1) *
      (game.blackDiscs - game.whiteDiscs), 0);
    return {
      ...player, played: played.length,
      wins: played.length ? wins : player.initialWins,
      draws, losses: played.length ? played.length - wins - draws : player.initialLosses,
      difference,
    };
  }).sort((a, b) => (a.rank ?? Infinity) - (b.rank ?? Infinity) || Number(a.id) - Number(b.id));
}

export function matchup(games, row, column) {
  const pair = games.filter(game => (game.black === row && game.white === column) ||
    (game.black === column && game.white === row));
  const wins = pair.filter(game => game.winner === row).length;
  const draws = pair.filter(game => game.winner === "draw").length;
  return { played: pair.length, wins, draws, losses: pair.length - wins - draws };
}
