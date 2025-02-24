"use client";

import React from "react";
import Header from "../components/Header";
import { Clock, MapPin } from "lucide-react";
import * as NBAIcons from "react-nba-logos";
import { Switch } from "@/components/ui/switch";

export default function Home() {
  const [liveBets, setLiveBets] = React.useState(false);

  return (
    <div className="w-full min-h-screen flex flex-col bg-gray-100">
      <Header />

      <div className="flex-1 w-full max-w-6xl mx-auto p-4 flex gap-6">
        {/* Left Section - All Games */}
        <div className="flex-1">
          {/* Top row: Tabs + Live Bets toggle */}
          <div className="flex items-center justify-between mb-6">
            {/* Tabs */}
            <div className="flex gap-6">
              <span className="font-medium border-b-2 border-red-500 pb-1">Featured</span>
              <span className="text-gray-500 cursor-pointer">Today</span>
              <span className="text-gray-500 cursor-pointer">Tomorrow</span>
              <span className="text-gray-500 cursor-pointer">Upcoming</span>
            </div>
            {/* Live Bets Toggle */}
            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-600">Live Bets</span>
              <Switch checked={liveBets} onCheckedChange={setLiveBets} />
            </div>
          </div>

          {/* Game Cards */}
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-white rounded-lg p-5 shadow-md">
              <div className="flex items-center gap-3">
                {NBAIcons.DET && <NBAIcons.DET size={36} />}
                <span className="font-semibold text-lg text-gray-900 font-inter">Pistons</span>
              </div>
              <div className="flex items-center gap-3 mt-2">
                {NBAIcons.MIN && <NBAIcons.MIN size={36} />}
                <span className="font-semibold text-lg text-gray-900 font-inter">Timberwolves</span>
              </div>

              <div className="mt-5 pt-4 border-t flex items-center justify-between">
                <div className="flex items-center gap-2 text-gray-500">
                  <MapPin className="h-5 w-5" />
                  <span className="text-sm font-medium font-inter">Detroit</span>
                </div>
                <div className="flex items-center gap-2 text-gray-700">
                  <Clock className="h-5 w-5" />
                  <span className="font-semibold text-sm font-inter">7:00 PM</span>
                </div>
              </div>
            </div>      


            {/* Dallas vs LA */}
            <div className="bg-white rounded-lg p-5 shadow-md">
              <div className="flex items-center gap-3">
                {NBAIcons.DAL && <NBAIcons.DAL size={36} />}
                <span className="font-semibold text-lg text-gray-900 font-inter">Mavericks</span>
              </div>
              <div className="flex items-center gap-3 mt-2">
                {NBAIcons.LAL && <NBAIcons.LAL size={36} />}
                <span className="font-semibold text-lg text-gray-900 font-inter">Lakers</span>
              </div>

              <div className="mt-5 pt-4 border-t flex items-center justify-between">
                <div className="flex items-center gap-2 text-gray-500">
                  <MapPin className="h-5 w-5" />
                  <span className="text-sm font-medium font-inter">Detroit</span>
                </div>
                <div className="flex items-center gap-2 text-gray-700">
                  <Clock className="h-5 w-5" />
                  <span className="font-semibold text-sm font-inter">7:00 PM</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Section */}
        <div className="w-72">
          <div className="bg-white rounded-lg p-4 mb-4">
            <h3 className="text-lg font-medium mb-3">Favorites</h3>
            <div className="bg-gray-100 rounded p-2">
              <p>Detroit vs. Minnesota</p>
            </div>
          </div>

          <div className="bg-white rounded-lg p-4">
            <h3 className="text-lg font-medium mb-3">Learn More</h3>
            <button className="w-full bg-blue-50 hover:bg-blue-100 p-2 rounded mb-2 text-left">
              What Does an Edge Mean in Terms of Betting?
            </button>
            <button className="w-full bg-red-50 hover:bg-red-100 p-2 rounded text-left">
              How to Read the Bet Website
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}