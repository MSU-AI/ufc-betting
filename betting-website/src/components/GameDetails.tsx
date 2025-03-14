"use client";
import React from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import OddsTable from "./OddsTable";
import Header from "./Header";
import { FaTrophy } from "react-icons/fa";
import {
  AlertTriangleIcon,
  ClockIcon,
  MapPinIcon,
  TrendingUpIcon,
} from "lucide-react";

export type OddsRow = {
  book: string;
  moneyline: string;
  probability: string;
  edge: string;
};

export type GameDetailsProps = {
  teamNames: string[];
  oddsData: {
    [teamName: string]: OddsRow[];
  };
  logos?: {
    [teamName: string]: string; // mapping from team name to full logo path
  };
};

// Fallback function in case a mapping isn’t provided
const getTeamLogo = (team: string) => `/logos/${team}.svg`;

function GameDetails({ teamNames, oddsData, logos }: GameDetailsProps) {
  // Determine the team with the highest winning probability from the first odds row
  const bestTeam = React.useMemo(() => {
    return teamNames.reduce((prev, curr) => {
      const currProb =
        parseFloat(oddsData[curr][0]?.probability.replace("%", "")) || 0;
      const prevProb =
        parseFloat(oddsData[prev][0]?.probability.replace("%", "")) || 0;
      return currProb > prevProb ? curr : prev;
    }, teamNames[0]);
  }, [teamNames, oddsData]);

  // Use logos prop if provided, otherwise fallback to the default function.
  const logoLeft = logos?.[teamNames[0]] ?? getTeamLogo(teamNames[0]);
  const logoRight = logos?.[teamNames[1]] ?? getTeamLogo(teamNames[1]);

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900">
      <Header />

      <main className="mx-auto px-4 sm:px-6 lg:px-8 mt-4 max-w-4xl">
        {/* Main card containing matchup info and team selection */}
        <div className="bg-white rounded-2xl shadow-lg p-4 sm:p-8">
          {/* Matchup heading with logos in a flex container */}
          <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold text-center -mb-3">
            <div className="flex items-center justify-center gap-4">
              {/* Render left team logo */}
              <div className="flex items-center justify-center">
                <img
                  src={logoLeft}
                  alt={`${teamNames[0]} logo`}
                  width={90}
                  height={90}
                />
              </div>
              <span>
                {teamNames[0]} vs. {teamNames[1]}
              </span>
              {/* Render right team logo */}
              <div className="flex items-center justify-center">
                <img
                  src={logoRight}
                  alt={`${teamNames[1]} logo`}
                  width={90}
                  height={90}
                />
              </div>
            </div>
          </h1>

          {/* Info bar with game details */}
          <div className="mb-4 space-y-3">
            <div className="flex items-center justify-center gap-3 text-sm sm:text-base text-gray-600">
              <div className="flex items-center gap-1.5 bg-gray-100 px-3 py-1 rounded-full">
                <ClockIcon className="h-4 w-4" />
                <span>7:00 PM ET</span>
              </div>
              <div className="flex items-center gap-1.5 bg-gray-100 px-3 py-1 rounded-full">
                <MapPinIcon className="h-4 w-4" />
                <span>Staples Center</span>
              </div>
            </div>

            <div className="flex flex-wrap items-center justify-center gap-x-4 gap-y-2 text-sm sm:text-base">
              <div className="flex items-center gap-1">
                <span className="font-medium text-purple-600">H2H:</span>
                <span>DET 3-1</span>
              </div>
              <div className="flex items-center gap-1">
                <TrendingUpIcon className="h-4 w-4 text-green-600" />
                <span>Over 220.5</span>
              </div>
              <div className="flex items-center gap-1 text-red-600">
                <AlertTriangleIcon className="h-4 w-4" />
                <span>Player X Out</span>
              </div>
            </div>
          </div>

          {/* Trophy display */}
          <div className="flex font-inter items-center justify-center gap-2 text-base sm:text-lg text-gray-800 mb-4">
            <FaTrophy className="text-yellow-500 h-4 w-4" />
            <span className="font-bold">{bestTeam}</span>
            <span className="text-xs sm:text-base font-normal">
              ({oddsData[bestTeam][0]?.probability ?? "X%"} probability)
            </span>
          </div>

          <Tabs defaultValue={teamNames[0]}>
            <TabsList className="flex justify-center space-x-1 mb-4">
              {teamNames.map((team) => (
                <TabsTrigger key={team} value={team}>
                  {team}
                </TabsTrigger>
              ))}
            </TabsList>

            {teamNames.map((team) => (
              <TabsContent key={team} value={team}>
                <div className="overflow-x-auto">
                  <OddsTable oddsData={oddsData[team]} />
                </div>
              </TabsContent>
            ))}
          </Tabs>
        </div>

        <div className="mt-6 p-4 bg-gradient-to-r from-blue-200 to-blue-100 border border-blue-300 rounded-lg shadow-lg flex items-center justify-center">
          <p className="text-center text-lg sm:text-xl font-bold text-blue-800">
            AI Prediction: <span className="underline">[Casino Name]</span> is most likely to win!
          </p>
        </div>
      </main>
    </div>
  );
}

export default GameDetails;
