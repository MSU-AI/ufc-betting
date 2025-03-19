import { Clock, MapPin, Calendar } from "lucide-react";
import { connectToDatabase } from "@/lib/mongodb";
import { getTeamLogo } from "@/lib/teamNameMap";
import Header from "../components/Header";
import Link from "next/link";
import { formatInTimeZone } from "date-fns-tz";
import { addDays } from "date-fns";

export default async function Home({
  searchParams,
}: {
  searchParams: { [key: string]: string | string[] | undefined };
}) {
  const sp = await Promise.resolve(searchParams);
  const { tab } = sp;
  const activeTab = Array.isArray(tab) ? tab[0] || "Featured" : tab || "Featured";
  const { db } = await connectToDatabase();
  const timeZone = "America/New_York";
  const now = new Date();
  const startTodayStr = formatInTimeZone(now, timeZone, "yyyy-MM-dd'T'00:00:00XXX");
  const endTodayStr = formatInTimeZone(now, timeZone, "yyyy-MM-dd'T'23:59:59XXX");
  const tomorrow = addDays(now, 1);
  const startTomorrowStr = formatInTimeZone(tomorrow, timeZone, "yyyy-MM-dd'T'00:00:00XXX");
  const endTomorrowStr = formatInTimeZone(tomorrow, timeZone, "yyyy-MM-dd'T'23:59:59XXX");
  const sevenDaysLater = addDays(now, 7);

  let games: any[] = [];
  if (activeTab === "Featured") {
    // Group documents by home_team, away_team, and commence_time
    games = await db.collection("ev_results")
      .aggregate([
        {
          $group: {
            _id: {
              home_team: "$home_team",
              away_team: "$away_team",
              commence_time: "$commence_time",
            },
            // Pick the first document in each group (you can adjust this if needed)
            doc: { $first: "$$ROOT" },
          },
        },
        { $replaceRoot: { newRoot: "$doc" } },
        { $sort: { commence_time: 1 } },
      ])
      .toArray();
    } else {
    // For other tabs, fetch from upcoming_games
    let query = {};
    if (activeTab === "Today") {
      query = {
        game_time: {
          $gte: new Date(startTodayStr).toISOString(),
          $lte: new Date(endTodayStr).toISOString(),
        },
      };
    } else if (activeTab === "Tomorrow") {
      query = {
        game_time: {
          $gte: new Date(startTomorrowStr).toISOString(),
          $lte: new Date(endTomorrowStr).toISOString(),
        },
      };
    } else if (activeTab === "Upcoming") {
      query = {
        game_time: {
          $gte: now.toISOString(),
          $lte: sevenDaysLater.toISOString(),
        },
      };
    }
    games = await db.collection("upcoming_games").find(query).sort({ game_time: 1 }).toArray();
  }

  return (
    <div className="w-full min-h-screen flex flex-col bg-gray-100">
      <Header />
      <div className="flex-1 w-full max-w-6xl mx-auto p-4 flex gap-6">
        <div className="flex-1">
          <div className="flex items-center justify-between mb-6">
            <div className="flex gap-6">
              {["Featured", "Today", "Tomorrow", "Upcoming"].map((tabName) => (
                <Link
                  key={tabName}
                  href={`/?tab=${tabName}`}
                  className={`font-medium pb-1 border-b-2 ${
                    activeTab === tabName ? "border-red-500" : "border-transparent text-gray-500"
                  }`}
                >
                  {tabName}
                </Link>
              ))}
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4">
            {games.length > 0 ? (
              games.map((game) => {
                const gameTime = new Date(game.commence_time || game.game_time);
                const home_team = game.home_team;
                const away_team = game.away_team;
                const gameId = game._id.toString();

                return (
                  <Link key={gameId} href={`/gamedetails?id=${encodeURIComponent(gameId)}`}>
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
                        <span className="font-semibold text-lg text-gray-900">{home_team}</span>
                      </div>
                      <div className="flex items-center gap-3 mt-3">
                        <div className="w-12 h-12 rounded-full overflow-hidden bg-white border border-gray-200">
                          <img
                            src={getTeamLogo(away_team)}
                            alt={`${away_team} logo`}
                            className="object-contain w-full h-full p-1"
                          />
                        </div>
                        <span className="font-semibold text-lg text-gray-900">{away_team}</span>
                      </div>
                      <div className="mt-5 pt-4 border-t flex items-center justify-between">
                        <div className="flex items-center gap-2 text-gray-500">
                          <MapPin className="h-5 w-5" />
                          <span className="text-sm font-medium">{game.arena || "TBD"}</span>
                        </div>
                        <div className="flex items-center gap-2 text-gray-700">
                          <Clock className="h-5 w-5" />
                          <span className="font-semibold text-sm">
                            {gameTime.toLocaleTimeString([], {
                              hour: "2-digit",
                              minute: "2-digit",
                              timeZoneName: "short",
                            })}
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
        </div>
      </div>
    </div>
  );
}