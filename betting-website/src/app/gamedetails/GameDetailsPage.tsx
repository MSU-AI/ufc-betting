import React from "react";
import GameDetails from "@/components/GameDetails";

// Using mock data until API is ready:
export default function GameDetailsPage() {
  const mockOddsData = {
    Detroit: [
      { book: "Caesar's Palace", moneyline: "+300", probability: "40%", edge: "-3%" },
      { book: "BetMGM", moneyline: "+250", probability: "20%", edge: "2.3%" },
      { book: "DraftKings", moneyline: "+300", probability: "40%", edge: "-3%" },
    ],
    Minnesota: [
      { book: "Caesar's Palace", moneyline: "+320", probability: "38%", edge: "-2.5%" },
      { book: "BetMGM", moneyline: "+280", probability: "42%", edge: "3%" },
      { book: "DraftKings", moneyline: "+310", probability: "40%", edge: "-2.8%" },
    ],
  };

  const teamLogos = {
    Detroit: "/logos/Detroit Pistons.svg",
    Minnesota: "/logos/Minnesota Timberwolves.svg",
  };

  const teamNames = Object.keys(mockOddsData);

  return (
    <GameDetails
      teamNames={teamNames}
      oddsData={mockOddsData}
      logos={teamLogos}
    />
  );
}
