/**
 * Merges arena (and potentially other details) from the upcoming_games collection
 * into a base game record from ev_results.
 * 
 * Finds games in upcoming_games with a game_time within 3 hours of commence_time.
 */
export async function mergeArenaInfo(baseGame: any, db: any): Promise<any> {
  const { home_team, away_team, commence_time } = baseGame;

  const baseGameTime = new Date(commence_time);
  const timeWindowMs = 3 * 60 * 60 * 1000; // +-3 hours
  const windowStart = new Date(baseGameTime.getTime() - timeWindowMs);
  const windowEnd = new Date(baseGameTime.getTime() + timeWindowMs);

  const match = await db.collection("upcoming_games").findOne({
    game_time: { $gte: windowStart.toISOString(), $lte: windowEnd.toISOString() },
    home_team: { $regex: home_team, $options: "i" },
    away_team: { $regex: away_team, $options: "i" }
  });

  if (match?.arena) {
    return { ...baseGame, arena: match.arena };
  }

  return { ...baseGame, arena: "Location not available" };
}
