"use client";
import React, { useState } from "react";
import { Inter } from "next/font/google"; // Import Inter from next/font

export type OddsRow = {
  book: string;
  moneyline: string;
  probability: string;
  edge: string;
  expectedValue: string;
};

export type GameDetailsProps = {
  teamNames: string[];
  oddsData: {
    [teamName: string]: OddsRow[];
  };
};

// we're using inter from figma!
const inter = Inter({
  subsets: ["latin"],
  display: "swap",
});

const GameDetails: React.FC<GameDetailsProps> = ({ teamNames, oddsData }) => {
  // starts with the first team in the list selected by default.
  const [selectedTeam, setSelectedTeam] = useState<string>(teamNames[0]);

  // get odds for the currently selected team.
  const displayedOdds = oddsData[selectedTeam];

  return (
    <div className={`${inter.className} min-h-screen bg-gray-50 text-gray-900`}>
      {/* header on the top */}
      <header className="bg-white p-4 shadow">
        <h1 className="text-xl font-bold">NBA Betting</h1>
      </header>

      {/* main content container */}
      <main className="max-w-5xl mx-auto mt-6 p-6 bg-white shadow rounded">
        {/* game title */}
        <h2 className="text-center text-2xl font-semibold mb-2">
          {teamNames.join(" vs. ")}
        </h2>
        <hr className="my-4 border-gray-200" />

        {/* team toggle buttons */}
        <div className="flex justify-center space-x-4">
          {teamNames.map((team) => (
            <button
              key={team}
              onClick={() => setSelectedTeam(team)}
              className={`px-6 py-2 rounded-full transition-colors border
                ${
                  selectedTeam === team
                    ? "bg-emerald-500 text-white border-emerald-500"
                    : "bg-white text-gray-700 border-gray-300 hover:bg-gray-100"
                }`}
            >
              {team}
            </button>
          ))}
        </div>

        {/* winner labels */}
        <p className="text-center mt-4 font-medium text-emerald-600">
          Most Likely Winner: {selectedTeam}!
        </p>

        {/* odds table */}
        <div className="mt-6 overflow-x-auto">
          <table className="w-full border-collapse text-left">
            <thead>
              <tr>
                <th className="py-3 px-4 text-sm font-semibold uppercase text-gray-500">
                  Book
                </th>
                <th className="py-3 px-4 text-sm font-semibold uppercase text-gray-500">
                  Moneyline
                </th>
                <th className="py-3 px-4 text-sm font-semibold uppercase text-gray-500">
                  Win Probability
                </th>
                <th className="py-3 px-4 text-sm font-semibold uppercase text-gray-500">
                  Edge
                </th>
                <th className="py-3 px-4 text-sm font-semibold uppercase text-gray-500">
                  Expected Value
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {displayedOdds?.map((row, idx) => (
                <tr key={idx} className="hover:bg-gray-50">
                  <td className="py-3 px-4">{row.book}</td>
                  <td className="py-3 px-4">{row.moneyline}</td>
                  <td className="py-3 px-4">{row.probability}</td>
                  <td className="py-3 px-4">{row.edge}</td>
                  <td className="py-3 px-4">{row.expectedValue}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </main>
    </div>
  );
};

export default GameDetails;
