"use client";
import { useEffect, useState } from "react";

interface GameResult {
  home_team: string;
  away_team: string;
  commence_time: string;
  home_win_prob: number;
  away_win_prob: number;
}

export default function GameResults() {
  const [results, setResults] = useState<GameResult[]>([]);

  useEffect(() => {
    fetch("/api/getResults")
      .then((res) => res.json())
      .then((data) => setResults(data))
      .catch((err) => console.error("Error fetching results:", err));
  }, []);

  return (
    <div>
      <h2>Game Results</h2>
      <ul>
        {results.map((game, index) => (
          <li key={index}>
            {game.home_team} vs {game.away_team} -{" "}
            {new Date(game.commence_time).toLocaleString()}
            <br />
            Home Win Probability: {game.home_win_prob.toFixed(2)}
            <br />
            Away Win Probability: {game.away_win_prob.toFixed(2)}
          </li>
        ))}
      </ul>
    </div>
  );
}
