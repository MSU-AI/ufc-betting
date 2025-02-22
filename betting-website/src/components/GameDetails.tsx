"use client";
import React, { useState } from "react";
import TeamToggle from "./TeamToggle";
import OddsTable from "./OddsTable";
import Header from "./Header";

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

const GameDetails: React.FC<GameDetailsProps> = ({ teamNames, oddsData }) => {
  const [selectedTeam, setSelectedTeam] = useState<string>(teamNames[0]);

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900">
      <Header />
    
      {/* content to display */}
      <main className="max-w-5xl mx-auto mt-6 p-6 bg-white shadow rounded">
        <h2 className="text-center text-2xl font-semibold mb-2">
          {teamNames.join(" vs. ")}
        </h2>
        <hr className="my-4 border-gray-200" />

        {/* toggle for display */}
        <TeamToggle teamNames={teamNames} selectedTeam={selectedTeam} setSelectedTeam={setSelectedTeam} />

        {/* winner announcement */}
        <p className="text-center mt-4 font-medium text-emerald-600">
          Most Likely Winner: {selectedTeam}!
        </p>

        {/* odds table */}
        <OddsTable oddsData={oddsData[selectedTeam]} />
      </main>
    </div>
  );
};

export default GameDetails;
