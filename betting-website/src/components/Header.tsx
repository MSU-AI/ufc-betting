"use client";
import React from "react";
import { Menu } from "lucide-react";

const Header: React.FC = () => {
  return (
    <header
      className="w-full border-b p-2 px-4 text-white shadow-md flex items-center justify-between"
      style={{ 
        fontFamily: "Barlow, sans-serif", 
        letterSpacing: "-0.11em",
        backgroundColor: "#C8102E"
      }}
    >
      <div className="flex items-center gap-2">
        <span className="text-2xl font-bold">NBA Betting</span>
      </div>
      <button className="flex items-center justify-center p-2 hover:bg-opacity-80 rounded">
        <Menu className="h-9 w-7" />
      </button>
    </header>
  );
};

export default Header;