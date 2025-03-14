import { connectToDatabase } from "@/lib/mongodb";

export default async function GameResults() {
  const { db } = await connectToDatabase();
  const results = await db.collection("games").find({}).toArray();

  return (
    <div className="p-4">
      <h2 className="text-xl font-bold mb-4">Game Results</h2>
      <ul className="space-y-3">
        {results.map((game) => (
          <li key={game._id.toString()} className="bg-white p-4 shadow rounded">
            <p className="text-lg font-semibold">
              {game.home_team} vs. {game.away_team}
            </p>
            <p className="text-sm text-gray-600">
              {new Date(game.commence_time).toLocaleString()}
            </p>
            <p className="text-sm">
              <span className="text-green-600">Home Win Probability:</span>{" "}
              {game.home_win_prob.toFixed(2)}
            </p>
            <p className="text-sm">
              <span className="text-red-600">Away Win Probability:</span>{" "}
              {game.away_win_prob.toFixed(2)}
            </p>
          </li>
        ))}
      </ul>
    </div>
  );
}
