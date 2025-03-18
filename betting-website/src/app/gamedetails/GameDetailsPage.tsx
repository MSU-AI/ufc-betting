// src/app/gamedetails/GameDetailsPage.tsx
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
  const { db } = await connectToDatabase();
  // If searchParams.id is an array, take the first element
  const idParam = searchParams.id;
  const id = Array.isArray(idParam) ? idParam[0] : idParam;

  if (!id) {
    return <p className="text-center text-gray-600">No game ID provided.</p>;
  }

  // Query by unique _id from ev_results
  const game = await db.collection("ev_results").findOne({ _id: new ObjectId(id) });
  if (!game) {
    return <p className="text-center text-gray-600">No game data available.</p>;
  }

  const { home_team, away_team, commence_time } = game;
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

  const formattedHomeWinProb =
    game.home_win_prob !== undefined ? `${(game.home_win_prob * 100).toFixed(2)}%` : "N/A";
  const formattedAwayWinProb =
    game.away_win_prob !== undefined ? `${(game.away_win_prob * 100).toFixed(2)}%` : "N/A";

  const oddsData = {
    [home_team]: game.bookmakers?.map((doc: any) => ({
      book: doc.bookmaker,
      moneyline: doc.home_odds,
      probability:
        doc.home_win_prob !== undefined
          ? `${(doc.home_win_prob * 100).toFixed(2)}%`
          : formattedHomeWinProb,
      edge: doc.home_ev !== undefined ? doc.home_ev.toFixed(2) : "N/A",
    })) || [],
    [away_team]: game.bookmakers?.map((doc: any) => ({
      book: doc.bookmaker,
      moneyline: doc.away_odds,
      probability:
        doc.away_win_prob !== undefined
          ? `${(doc.away_win_prob * 100).toFixed(2)}%`
          : formattedAwayWinProb,
      edge: doc.away_ev !== undefined ? doc.away_ev.toFixed(2) : "N/A",
    })) || [],
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
