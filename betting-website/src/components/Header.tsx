"use client";
import React from "react";
import { Menu } from "lucide-react";

const Header: React.FC = () => {
  return (
    <header
      className="w-full bg-red-600 border-b p-2 px-4 text-white shadow-md flex items-center justify-between"
      style={{ fontFamily: "Barlow, sans-serif", letterSpacing: "-0.05em" }}
    >
      <div className="flex items-center gap-2">
        <span className="text-base font-bold">NBA Betting</span>
      </div>
      <button className="flex items-center justify-center p-2 hover:bg-red-700 rounded">
        <Menu className="h-5 w-5" />
      </button>
    </header>
  );
};

export default Header;