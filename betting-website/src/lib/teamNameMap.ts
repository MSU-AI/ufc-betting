/**
 *  ⚠️ Note: modification of this existing functions may affect other parts of the codebase. 
 *  Please ensure to test thoroughly after any changes.
 * 
 * This module provides a mapping of NBA team acronyms to their full names and aliases.
 * It includes functions to:
 * - Convert between acronyms, aliases, and full names
 * - Retrieve team logos based on full team names (this will be referred to as canonical names throughout the documentation)
 *
 * All lookups are case-insensitive and work with common variations of team names.
 *
 * FUNCTIONALITY EXAMPLES 
 * 
 * acronym to full name:
 * convertTeamName("LAL")           "LAL"
 * getCanonicalTeamName("LAL")      "Los Angeles Lakers"
 * getTeamLogo("LAL")               "/logos/Los%20Angeles%20Lakers.svg"
 * 
 * alias to acronym:
 * convertTeamName("Lakers")        "LAL"
 * getCanonicalTeamName("Lakers")   "Los Angeles Lakers"
 * getTeamLogo("Lakers")            "/logos/Los%20Angeles%20Lakers.svg"
 * 
 * full name to acronym:
 * convertTeamName("Golden State Warriors")      "GSW"
 * getCanonicalTeamName("Golden State Warriors") "Golden State Warriors"
 * 
 * abbreviations/slang:
 * convertTeamName("76ers")          "PHI"
 * getCanonicalTeamName("76ers")     "Philadelphia 76ers"
 * 
 * sorta close matches:
 * convertTeamName("NY Knicks")      "NYK"
 * getCanonicalTeamName("NY Knicks") "New York Knicks"
 * 
 * short version of a name:
 * getShortTeamName("Los Angeles Lakers") "Lakers"
 * 
 * error handling:
 * convertTeamName("Some Unknown Team") 
 * Throws: "Could not convert team name or acronym: Some Unknown Team"
 * 
 */

export const teamAliases: Record<string, string[]> = {
  ATL: ["Atlanta Hawks", "Hawks"],
  BOS: ["Boston Celtics", "Celtics"],
  BRK: ["Brooklyn Nets", "Nets"],
  CHO: ["Charlotte Hornets", "Hornets"],
  CHI: ["Chicago Bulls", "Bulls"],
  CLE: ["Cleveland Cavaliers", "Cavaliers", "Cavs"],
  DAL: ["Dallas Mavericks", "Mavericks"],
  DEN: ["Denver Nuggets", "Nuggets"],
  DET: ["Detroit Pistons", "Pistons"],
  GSW: ["Golden State Warriors", "GS Warriors", "Golden State", "Warriors"],
  HOU: ["Houston Rockets", "Rockets"],
  IND: ["Indiana Pacers", "Pacers"],
  LAC: ["Los Angeles Clippers", "LA Clippers", "Clippers"],
  LAL: ["Los Angeles Lakers", "LA Lakers", "Lakers"],
  MEM: ["Memphis Grizzlies", "Grizzlies"],
  MIA: ["Miami Heat", "Heat"],
  MIL: ["Milwaukee Bucks", "Bucks"],
  MIN: ["Minnesota Timberwolves", "Timberwolves"],
  NOP: ["New Orleans Pelicans", "Pelicans"],
  NYK: ["New York Knicks", "NY Knicks", "Knicks"],
  OKC: ["Oklahoma City Thunder", "OKC Thunder", "Thunder"],
  ORL: ["Orlando Magic", "Magic"],
  PHI: ["Philadelphia 76ers", "Philly 76ers", "76ers", "Sixers"],
  PHO: ["Phoenix Suns", "Suns"],
  POR: ["Portland Trail Blazers", "Trail Blazers", "Blazers", "Portland Blazers"],
  SAC: ["Sacramento Kings", "Sacramento"],
  SAS: ["San Antonio Spurs", "San Antonio", "Spurs"],
  TOR: ["Toronto Raptors", "Toronto", "Raptors"],
  UTA: ["Utah Jazz", "Jazz"],
  WAS: ["Washington Wizards", "Washington", "Wizards"],
};


/**
 * Map of team acronyms to their full team names. 
 * The canonical name is the full name of the team, which is the first item in the alias array.
 * 
 * This is useful when you want to display or fetch resources based on a
 * standardized team name, regardless of the input format.
 *
 * Example: acronymToCanonical["GSW"] === "Golden State Warriors"
 */
export const acronymToCanonical: Record<string, string> = Object.keys(teamAliases).reduce((acc, code) => {
  acc[code] = teamAliases[code][0];
  return acc;
}, {} as Record<string, string>);


/**
 * Reverse lookup map from any lowercase alias
 * to its corresponding 3-letter acronym 
 *
 * This allows you to accept various user input formats or fuzzy sources
 * and normalize them back to consistent 3-letter codes.
 *
 * Example: aliasToAcronym["warriors"] === "GSW" 
 */
export const aliasToAcronym: Record<string, string> = {};
for (const [code, names] of Object.entries(teamAliases)) {
  for (const alias of names) {
    aliasToAcronym[alias.toLowerCase()] = code;
  }
}

/**
 * Converts a team name or alias to its three-letter acronym.
 * If the input is already a valid acronym, it is returned as is.
 * @param name - The team name, alias, or acronym.
 * @returns The three-letter acronym.
 * @throws Error if the input cannot be converted.
 */
export function convertTeamName(name: string): string {
  if (!name || typeof name !== "string") {
    throw new Error("Input must be a non-empty string");
  }
  const trimmed = name.trim();
  const upper = trimmed.toUpperCase();
  if (acronymToCanonical[upper]) {
    return upper;
  }
  const lower = trimmed.toLowerCase();
  if (aliasToAcronym[lower]) {
    return aliasToAcronym[lower];
  }
  throw new Error(`Could not convert team name or acronym: ${name}`);
}

/**
 * Given a team name, alias, or acronym, returns its canonical full name.
 * @param name - The team name, alias, or acronym.
 * @returns The canonical full team name.
 */
export function getCanonicalTeamName(name: string): string {
  const acronym = convertTeamName(name);
  return acronymToCanonical[acronym];
}


/**
 * Returns the path to the logo of a team based on its name or acronym.
 * @param team - The team name, alias, or acronym.
 * @returns The URL path to the team's logo.
 */
export const getTeamLogo = (team: string): string => {
  const canonicalTeam = getCanonicalTeamName(team);
  return `/logos/${encodeURIComponent(canonicalTeam)}.svg`;
}

/**
 * Returns the short version of a canonical team name
 * Falls back to the original name if no alias is found.
 *
 * @param name - The full/canonical team name, acronym, or alias.
 * @returns The short name like "Lakers", "Warriors", etc.
 */
export function getShortTeamName(name: string): string {
  try {
    const acronym = convertTeamName(name);
    const aliases = teamAliases[acronym];
    return aliases?.[aliases.length - 1] || name;
  } catch {
    return name; 
  }
}
