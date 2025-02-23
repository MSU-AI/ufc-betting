import GameDetails from "../../components/GameDetails";

// Uncomment when we're ready!
// const API_URL = "...";

// export default async function GameDetailsPage() {
//   // where we fetch data from the api and pass it into the details
//   const response = await fetch(API_URL);
//   // if needed, we can add headers (like responses)
//   if (!response.ok) {
//     console.error("Failed to fetch data from API");
//     return <div>Failed to fetch data.</div>
//   }

//   const oddsData = await response.json();
//   return <GameDetails oddsData={oddsData} />;
// }

// Using mock data until API is ready:
export default function GameDetailsPage() {
  // Mock data that supports dynamic team names
  const mockOddsData = {
    Detroit: [
      {
        book: "Caesar's Palace",
        moneyline: "+300",
        probability: "40%",
        edge: "3%",
        expectedValue: "$2",
      },
      {
        book: "BetMGM",
        moneyline: "+250",
        probability: "20%",
        edge: "2.3%",
        expectedValue: "$4",
      },
      {
        book: "DraftKings",
        moneyline: "+300",
        probability: "40%",
        edge: "3%",
        expectedValue: "$2",
      },
    ],
    Minnesota: [
      {
        book: "Caesar's Palace",
        moneyline: "+320",
        probability: "38%",
        edge: "2.5%",
        expectedValue: "$2.5",
      },
      {
        book: "BetMGM",
        moneyline: "+280",
        probability: "42%",
        edge: "3%",
        expectedValue: "$1.8",
      },
      {
        book: "DraftKings",
        moneyline: "+310",
        probability: "40%",
        edge: "2.8%",
        expectedValue: "$2.2",
      },
    ],
  };

  // dervies keys dynamically.. from the keys of mockOddsData.
  const teamNames = Object.keys(mockOddsData);
  

  return <GameDetails teamNames={teamNames} oddsData={mockOddsData} />;
}


