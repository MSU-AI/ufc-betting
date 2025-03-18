import React from "react";
import { ObjectId } from "mongodb";
import GameDetails from "@/components/GameDetails";
import { connectToDatabase } from "@/lib/mongodb";
import { getTeamLogo } from "@/lib/teamNameMap";

export default async function GameDetailsPage({
  searchParams,
}: {
  searchParams: { id?: string | string[] };
}) {
  const sp = await Promise.resolve(searchParams);
  const idParam = sp.id;
  const id = Array.isArray(idParam) ? idParam[0] : idParam;
  if (!id) {
    return <p className="text-center text-gray-600">No game ID provided.</p>;
  }
  const { db } = await connectToDatabase();

  // Fetch one record to use as a reference for grouping
  const baseGameRecord = await db.collection("ev_results").findOne({ _id: new ObjectId(id) });
  if (!baseGameRecord) {
    return <p className="text-center text-gray-600">No game data available.</p>;
  }

  const { home_team, away_team, commence_time } = baseGameRecord;
  // Query all records matching the same game (using home_team, away_team, and commence_time)
  const gameRecords = await db.collection("ev_results").find({ home_team, away_team, commence_time }).toArray();
  const bookmakers = gameRecords.map((record) => ({
    book: record.bookmaker,
    home_moneyline: record.home_odds.toString(),
    home_probability: `${(record.home_win_prob * 100).toFixed(2)}%`,
    home_edge: record.home_ev.toString(),
    away_moneyline: record.away_odds.toString(),
    away_probability: `${(record.away_win_prob * 100).toFixed(2)}%`,
    away_edge: record.away_ev.toString(),
  }));
  
  const oddsData = {
    [home_team]: bookmakers.map((b) => ({
      book: b.book,
      moneyline: b.home_moneyline,
      probability: b.home_probability,
      edge: b.home_edge,
    })),
    [away_team]: bookmakers.map((b) => ({
      book: b.book,
      moneyline: b.away_moneyline,
      probability: b.away_probability,
      edge: b.away_edge,
    })),
  };

  const eventTime = new Date(commence_time);
  const gameDetails = {
    game_time: eventTime.toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      timeZoneName: "short",
    }),
    arena: "Location not available",
    h2h_record: "H2H data not available",
    over_under: "Over/Under data not available",
    player_injury: "No injury updates",
  };

  const teamLogos = {
    [home_team]: getTeamLogo(home_team),
    [away_team]: getTeamLogo(away_team),
  };

  const teamNames = [home_team, away_team];
  
  return (
    <GameDetails
      teamNames={teamNames}
      oddsData={oddsData}
      logos={teamLogos}
      gameDetails={gameDetails}
    />
  );
}
