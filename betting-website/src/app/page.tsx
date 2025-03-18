"use client";

import Header from "../components/Header";

const tabs = ["Featured", "Today", "Tomorrow", "Upcoming"];

export default function Home() {
  return (
    <div className="min-h-screen bg-gray-100">
      <Header />
      <div className="max-w-6xl mx-auto p-4">
        <div className="flex space-x-2 border-b pb-2">
          {tabs.map((tab) => (
            <button
              key={tab}
              className="px-4 py-2 rounded bg-white"
            >
              {tab}
            </button>
          ))}
        </div>
        <div className="grid grid-cols-2 gap-4 mt-4">
          {[1, 2].map((_, index) => (
            <div key={index} className="bg-white p-4 rounded-lg shadow">
              <div className="flex items-center space-x-2">
                <div className="w-10 h-10 bg-gray-300 rounded-full"></div>
                <div className="font-bold">Detroit vs. Minnesota</div>
              </div>
              <div className="mt-2 text-gray-600">Location: Detroit City</div>
              <div className="text-right font-semibold">7:00 PM</div>
            </div>
          ))}
        </div>
      </div>
      <div className="max-w-sm mx-auto p-4 mt-6 bg-white shadow-lg rounded-lg">
        <h2 className="font-bold mb-2">Favorites</h2>
        <div className="bg-gray-200 p-2 rounded">Detroit vs. Minnesota</div>
      </div>
      <div className="max-w-sm mx-auto p-4 mt-6 bg-white shadow-lg rounded-lg">
        <h2 className="font-bold mb-2">Learn More</h2>

        <button className="w-full bg-black text-white p-2 rounded mb-2">
          What does an Edge Mean in terms of betting?
        </button>
        <button className="w-full bg-black text-white p-2 rounded">
          How to Read the Bet Website
        </button>
      </div>
    </div>
  );
}