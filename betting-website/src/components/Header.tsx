"use client";
import React from "react";
import Link from "next/link";
import { Menu } from "lucide-react";
import { motion } from "framer-motion";
import HeaderAnimation from "./HeaderAnimation";

export interface HeaderProps {
  activeTab?: string;
  iversonText?: string;
  iversonStyle?: React.CSSProperties;
  backgroundColor?: string;
  animationOffsetY?: number;
}

export default function Header({
  activeTab,
  iversonText = "Iverson",
  iversonStyle,
  backgroundColor = "#C8102E",
  animationOffsetY = -42,
}: HeaderProps) {
  return (
    <header
      className="w-full shadow-md flex flex-col font-nfl"
      style={{ letterSpacing: "0.25em" }}
    >
      <div 
        className="w-full p-5 px-8 border-b border-white/10" 
        style={{ backgroundColor }}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 relative">
            {/* HeaderAnimation receives the offset prop */}
            <HeaderAnimation 
              animDuration={2.5} 
              offsetY={animationOffsetY} 
            />
            {/* Iverson text is managed here */}
            <motion.span
              className="text-6xl font-bold relative z-20"
              style={{
                background: "linear-gradient(to bottom, #FFFFFF 15%, #E2E8F0 60%, #A1A1AA 95%)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
                textShadow: "0 1px 3px rgba(0,0,0,0.25)",
                ...iversonStyle
              }}
              animate={{
                x: [0, 30, 30, 0],
                scale: [1, 1.08, 1.08, 1],
                filter: [
                  "drop-shadow(0 0 0 rgba(255,255,255,0))",
                  "drop-shadow(0 0 12px rgba(255,255,255,0.75))",
                  "drop-shadow(0 0 12px rgba(255,255,255,0.75))",
                  "drop-shadow(0 0 0 rgba(255,255,255,0))",
                ],
                transition: {
                  duration: 8,
                  x: { duration: 8, times: [0, 0.2, 0.8, 1], ease: "easeInOut" },
                  scale: { duration: 8, times: [0, 0.2, 0.8, 1], ease: "easeInOut" },
                  filter: { duration: 8, times: [0, 0.2, 0.8, 1], ease: "easeInOut" },
                },
              }}
            >
              {iversonText}
            </motion.span>
          </div>
          <motion.button
            className="flex items-center justify-center p-2 rounded"
            whileHover={{ scale: 1.1, backgroundColor: "rgba(255,255,255,0.2)" }}
            whileTap={{ scale: 0.95 }}
          >
            {/* Hamburger menu we can use later if we need to */}
            {/* <Menu className="h-11 w-11 text-white" /> */}
          </motion.button>
        </div>
      </div>

      {/* Navigation tabs section with darker red background */}
      {activeTab && (
        <div className="bg-[#B7112A] text-white py-2">
          <nav className="px-8">
            <ul className="flex justify-center gap-6 sm:gap-8">
              {["Featured", "Today", "Tomorrow", "Upcoming"].map((tabName) => (
                <li key={tabName} className="relative">
                  <Link
                    href={`/?tab=${tabName}`}
                    className={`
                      uppercase tracking-[0.15em] text-lg md:text-xl font-extrabold py-1.5 px-1 inline-block text-center
                      font-['Bebas_Neue',_sans-serif] transition-colors duration-200
                      ${activeTab === tabName ? "text-amber-400" : "text-gray-200 hover:text-white"}
                    `}
                  >
                    {tabName}
                    <div 
                      className={`
                        absolute bottom-0 left-1/2 transform -translate-x-1/2 w-full h-[3px] 
                        ${activeTab === tabName ? "bg-amber-400" : "bg-transparent"}
                      `}
                      style={{ width: `calc(100% - 4px)` }}
                    />
                  </Link>
                </li>
              ))}
            </ul>
          </nav>
        </div>
      )}
    </header>
  );
}
