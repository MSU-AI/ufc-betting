import { Clock, MapPin, Calendar } from "lucide-react";
import { connectToDatabase } from "@/lib/mongodb";
import { getTeamLogo } from "@/lib/teamNameMap";
import Header from "../components/Header";
import Link from "next/link";
import { formatInTimeZone } from "date-fns-tz";
import { addDays } from "date-fns";
import { mergeArenaInfo } from "@/lib/mergeGameData"; 
import GamesGrid from "@/components/GamesGrid";
import TimeZoneSync from '@/components/utils/TimeZoneSync';

export default async function Home({
  searchParams,
}: {
  searchParams: Promise<{ [key: string]: string | string[] | undefined }>;
}) {
  const sp = await searchParams;
  const { tab, tz } = sp;
  const { db } = await connectToDatabase();

  const activeTab = Array.isArray(tab) ? tab[0] || "Featured" : tab || "Featured";
  const timeZone = Array.isArray(tz) ? tz[0] : tz || "America/New_York";
  const now = new Date();
  const startTodayStr = formatInTimeZone(now, timeZone, "yyyy-MM-dd'T'00:00:00XXX");
  const endTodayStr = formatInTimeZone(now, timeZone, "yyyy-MM-dd'T'23:59:59XXX");
  const tomorrow = addDays(now, 1);
  const startTomorrowStr = formatInTimeZone(tomorrow, timeZone, "yyyy-MM-dd'T'00:00:00XXX");
  const endTomorrowStr = formatInTimeZone(tomorrow, timeZone, "yyyy-MM-dd'T'23:59:59XXX");
  const sevenDaysLater = addDays(now, 7);

  let games: any[] = [];

  if (activeTab === "Featured") {
    // Only include games on the current day (from now until end of today)
    games = await db.collection("ev_results")
      .aggregate([
        {
          $match: {
            $expr: {
              $and: [
                { $gte: [ "$commence_time", now ] },
                { $lte: [ "$commence_time", new Date(endTodayStr) ] }
              ]
            }
          }
        },
        {
          $group: {
            _id: {
              home_team: "$home_team",
              away_team: "$away_team",
              commence_time: "$commence_time",
            },
            doc: { $first: "$$ROOT" },
            maxHomeEV: { $max: "$home_ev" },
            maxAwayEV: { $max: "$away_ev" },
          },
        },
        {
          $addFields: {
            maxPositiveEV: {
              $cond: [
                { $gt: [{ $max: ["$maxHomeEV", "$maxAwayEV"] }, 0] },
                { $max: ["$maxHomeEV", "$maxAwayEV"] },
                -1,
              ],
            },
          },
        },
        { $match: { maxPositiveEV: { $gt: 0 } } },
        { $sort: { maxPositiveEV: -1 } },
        { $limit: 4 },
        { $replaceRoot: { newRoot: "$doc" } },
      ])
      .toArray();
  } else if (activeTab === "Today") {
    games = await db.collection("ev_results")
      .aggregate([
        {
          $match: {
            $expr: {
              $and: [
                { $gte: [ "$commence_time", new Date(startTodayStr) ] },
                { $lte: [ "$commence_time", new Date(endTodayStr) ] }
              ]
            }
          }
        },
        {
          $group: {
            _id: {
              home_team: "$home_team",
              away_team: "$away_team",
              commence_time: "$commence_time",
            },
            doc: { $first: "$$ROOT" },
          },
        },
        { $replaceRoot: { newRoot: "$doc" } },
        { $sort: { commence_time: 1 } },
      ])
      .toArray();
  } else if (activeTab === "Tomorrow") {
    games = await db.collection("ev_results")
      .aggregate([
        {
          $match: {
            $expr: {
              $and: [
                { $gte: [ "$commence_time", new Date(startTomorrowStr) ] },
                { $lte: [ "$commence_time", new Date(endTomorrowStr) ] }
              ]
            }
          }
        },
        {
          $group: {
            _id: {
              home_team: "$home_team",
              away_team: "$away_team",
              commence_time: "$commence_time",
            },
            doc: { $first: "$$ROOT" },
          },
        },
        { $replaceRoot: { newRoot: "$doc" } },
        { $sort: { commence_time: 1 } },
      ])
      .toArray();
  } else if (activeTab === "Upcoming") {
    games = await db.collection("ev_results")
      .aggregate([
        {
          $match: {
            $expr: {
              $and: [
                { $gte: [ "$commence_time", now ] },
                { $lte: [ "$commence_time", sevenDaysLater ] }
              ]
            }
          }
        },
        {
          $group: {
            _id: {
              home_team: "$home_team",
              away_team: "$away_team",
              commence_time: "$commence_time",
            },
            doc: { $first: "$$ROOT" },
          },
        },
        { $replaceRoot: { newRoot: "$doc" } },
        { $sort: { commence_time: 1 } },
      ])
      .toArray();
  }

  games = await Promise.all(games.map(async (game) => {
    return await mergeArenaInfo(game, db);
  }));
  games = JSON.parse(JSON.stringify(games)); // for json serialization

  return (
    <div className="w-full min-h-screen flex flex-col bg-gray-100">
      <TimeZoneSync />
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
                    activeTab === tabName
                      ? "border-red-500"
                      : "border-transparent text-gray-500"
                  }`}
                >
                  {tabName}
                </Link>
              ))}
            </div>
          </div>

          <GamesGrid games={games} activeTab={activeTab} />
        </div>
      </div>
    </div>
  );

}
