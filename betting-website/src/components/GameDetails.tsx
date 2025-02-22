"use client";
import React from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
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
  return (
    <div className="min-h-screen bg-white text-gray-900">
      {/* Teal header remains for branding */}
      <Header />

      <main className="mx-auto mt-8 px-4 sm:px-6 lg:px-8">
        {/* White card container with subtle shadow */}
        <div className="max-w-4xl mx-auto bg-white shadow-md rounded-lg p-6">
          <h2 className="text-center text-2xl font-semibold mb-2">
            {teamNames.join(" vs. ")}
          </h2>
          <hr className="my-4 border-gray-200" />

          <Tabs defaultValue={teamNames[0]}>
            <TabsList className="flex justify-center mb-4 space-x-4">
              {teamNames.map((team) => (
                <TabsTrigger
                  key={team}
                  value={team}
                  className="px-4 py-2 text-sm font-medium text-gray-600 transition-colors duration-150 hover:text-gray-800data-[state=active]:text-gray-900 data-[state=active]:border-b-2 data-[state=active]:border-gray-900"
                >
                  {team}
                </TabsTrigger>
              ))}
            </TabsList>

            {teamNames.map((team) => (
              <TabsContent key={team} value={team}>
                <OddsTable oddsData={oddsData[team]} />
                <p className="text-center mt-4 text-gray-700">
                 <span className="inline-flex items-center justify-center gap-1">
                   <span className="text-teal-600 font-semibold">{team}!</span>
                   <span className="text-teal-600">is the likely winner ✅</span>
                 </span>
                </p>
              </TabsContent>
            ))}
          </Tabs>
        </div>
      </main>
    </div>
  );
};

export default GameDetails;
