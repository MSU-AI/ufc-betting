import Image from "next/image";
import Header from "../components/Header";

export default function Home() {
  return (
    <div className="w-full h-screen flex flex-col">
      {/* Header */}
      <Header />

      {/* Main Content */}
      <div className="flex flex-1 w-full max-w-6xl mx-auto mt-4 p-4 gap-4">
        {/* Left Section - All Games */}
        <div className="flex-1 bg-white shadow-md p-4 rounded-md border border-blue-500">
          <h2 className="text-center text-xl font-semibold">All Games</h2>
          <div className="flex justify-center gap-4 border-b pb-2 mt-2">
            <span className="font-bold">Featured</span>
            <span className="text-gray-500">Today</span>
            <span className="text-gray-500">Tomorrow</span>
            <span className="text-gray-500">Upcoming</span>
          </div>
          
          {/* Game Cards */}
          <div className="grid grid-cols-2 gap-4 mt-4">
            {[1, 2].map((game) => (
              <div key={game} className="p-4 bg-white shadow-md rounded-md">
                <h3 className="text-lg font-semibold">Detroit vs Minnesota</h3>
                <p className="text-gray-500">Location: Detroit City</p>
                <p className="text-right font-bold">7:00 PM</p>
              </div>
            ))}
          </div>
        </div>

        {/* Right Section - Favorites & Learn More */}
        <div className="w-1/4 flex flex-col gap-4">
          {/* Favorites */}
          <div className="bg-white p-4 shadow-md rounded-md">
            <h3 className="text-lg font-semibold">Favorites</h3>
            <div className="p-2 bg-gray-200 rounded-md mt-2">Detroit vs Minnesota</div>
          </div>
          
          {/* Learn More */}
          <div className="bg-white p-4 shadow-md rounded-md">
            <h3 className="text-lg font-semibold">Learn More</h3>
            <button className="w-full bg-blue-300 p-2 rounded-md mt-2 text-left">
              What Does an Edge Mean in Terms of Betting?
            </button>
            <button className="w-full bg-red-300 p-2 rounded-md mt-2 text-left">
              How to Read the Bet Website
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
