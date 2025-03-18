"use client";
import React from "react";
import { Menu } from "lucide-react";

const Header: React.FC = () => {
  return (
    <header
      className="w-full border-b p-6 px-8 text-white shadow-md flex items-center justify-between bg-[#C8102E] font-nfl"
      style={{ letterSpacing: "0.25em" }}
    >
      <div className="flex items-center gap-2">
        <span className="text-5xl font-bold">Iverson</span>
      </div>
      <button className="flex items-center justify-center p-2 hover:bg-opacity-80 rounded">
        <Menu className="h-10 w-10" />
      </button>
    </header>
  );
};


export default Header;