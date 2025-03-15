import React from "react";
import GameDetails from "@/components/GameDetails";
import { connectToDatabase } from "@/lib/mongodb";
import { getTeamLogo } from "@/lib/teamNameMap";

export default async function GameDetailsPage() {
  const { db } = await connectToDatabase();
  const groupedGames = await db
    .collection("ev_results")
    .aggregate([
      {
        $group: {
          _id: {
            home_team: "$home_team",
            away_team: "$away_team",
            commence_time: "$commence_time",
          },
          home_code: { $first: "$home_code" },
          away_code: { $first: "$away_code" },
          home_win_prob: { $first: "$home_win_prob" },
          away_win_prob: { $first: "$away_win_prob" },
          bookmakers: {
            $push: {
              bookmaker: "$bookmaker",
              home_odds: "$home_odds",
              away_odds: "$away_odds",
              home_ev: "$home_ev",
              away_ev: "$away_ev",
            },
          },
        },
      },
      { $sort: { "_id.commence_time": 1 } },
      { $limit: 1 } // Only take the earliest upcoming event from ev_results FOR NOW
    ])
    .toArray();


  if (!groupedGames.length) {
    return <p className="text-center text-gray-600">No game data available.</p>;
  }

  const game = groupedGames[0];
  const { home_team, away_team, commence_time } = game._id;
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
    [home_team]: game.bookmakers.map((doc: any) => ({
      book: doc.bookmaker,
      moneyline: doc.home_odds,
      probability:
        doc.home_win_prob !== undefined ? `${(doc.home_win_prob * 100).toFixed(2)}%` : formattedHomeWinProb,
      edge: doc.home_ev !== undefined ? doc.home_ev.toFixed(2) : "N/A",
    })),
    [away_team]: game.bookmakers.map((doc: any) => ({
      book: doc.bookmaker,
      moneyline: doc.away_odds,
      probability:
        doc.away_win_prob !== undefined ? `${(doc.away_win_prob * 100).toFixed(2)}%` : formattedAwayWinProb,
      edge: doc.away_ev !== undefined ? doc.away_ev.toFixed(2) : "N/A",
    })),
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
