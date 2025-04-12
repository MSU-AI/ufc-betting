"use client";

import React from "react";
import Link from "next/link";
import { Clock, MapPin, Calendar } from "lucide-react";
import { formatInTimeZone } from "date-fns-tz";
import { getTeamLogo } from "@/lib/teamNameMap";
import { useUserTimeZone } from "@/lib/timeZone";

type GamesGridProps = {
  games: any[];
  activeTab: string;
};

export default function GamesGrid({ games, activeTab }: GamesGridProps) {
  const userTimeZone = useUserTimeZone();

  return (
    <div className="grid grid-cols-2 gap-4">
      {games.length > 0 ? (
        games.map((game) => {
          const gameTime = new Date(game.commence_time);
          const home_team = game.home_team;
          const away_team = game.away_team;
          const gameId = game._id.toString();

          return (
            <Link key={gameId} href={`/gamedetails/${gameId}`}>
              <div className="bg-white rounded-lg p-5 shadow-md cursor-pointer">
                <div className="flex items-center gap-2 text-gray-700 mb-4">
                  <Calendar className="h-5 w-5 text-gray-500" />
                  <span className="font-semibold text-md">
                    {gameTime.toLocaleDateString(undefined, {
                      weekday: "long",
                      year: "numeric",
                      month: "long",
                      day: "numeric",
                    })}
                  </span>
                </div>
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 rounded-full overflow-hidden bg-white border border-gray-200">
                    <img
                      src={getTeamLogo(home_team)}
                      alt={`${home_team} logo`}
                      className="object-contain w-full h-full p-1"
                    />
                  </div>
                  <span className="font-semibold text-lg text-gray-900">
                    {home_team}
                  </span>
                </div>
                <div className="flex items-center gap-3 mt-3">
                  <div className="w-12 h-12 rounded-full overflow-hidden bg-white border border-gray-200">
                    <img
                      src={getTeamLogo(away_team)}
                      alt={`${away_team} logo`}
                      className="object-contain w-full h-full p-1"
                    />
                  </div>
                  <span className="font-semibold text-lg text-gray-900">
                    {away_team}
                  </span>
                </div>
                <div className="mt-5 pt-4 border-t flex items-center justify-between">
                  <div className="flex items-center gap-2 text-gray-500">
                    <MapPin className="h-5 w-5" />
                    <span className="text-sm font-medium">
                      {game.arena || "TBD"}
                    </span>
                  </div>
                  <div className="flex items-center gap-2 text-gray-700">
                    <Clock className="h-5 w-5" />
                    <span className="font-semibold text-sm">
                      {formatInTimeZone(gameTime, userTimeZone, 'h:mm a zzz')}
                    </span>
                  </div>
                </div>
              </div>
            </Link>
          );
        })
      ) : (
        <p className="text-gray-600 text-center col-span-2">
          No games available for {activeTab}.
        </p>
      )}
    </div>
  );
}
