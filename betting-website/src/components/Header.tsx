"use client";
import React from "react";
import { Menu } from "lucide-react";


export default function Header() {
  return (
    // <header
    //   className="w-full border-b p-6 px-8 text-white shadow-md flex items-center justify-between bg-[#C8102E] font-nfl relative"
    //   style={{ letterSpacing: "0.25em" }}
    // >
    //   <div className="w-12 h-12"></div>
    //   <div className="absolute left-0 right-0 flex justify-center items-center pointer-events-none">
    //     <span className="text-7xl font-bold">Iverson</span>
    //   </div>
    //   <button className="flex items-center justify-center p-3 hover:bg-white/20 rounded-lg transition-colors z-10">
    //     <Menu className="h-12 w-12" />
    //   </button>
    //</header>
    <header
      className="w-full border-b p-5 px-8 text-white shadow-md flex items-center justify-between bg-[#C8102E] font-nfl"
      style={{ letterSpacing: "0.25em" }}
    >
      <div className="flex items-center gap-2">
        <span className="text-6xl font-bold">Iverson</span>
      </div>
      <button className="flex items-center justify-center p-2 hover:bg-opacity-80 rounded">
        <Menu className="h-11 w-11" />
      </button>
    </header>

  );
}
// const Header: React.FC = () => {
//   return (
//     <header
//       className="w-full border-b p-6 px-8 text-white shadow-md flex items-center justify-between bg-[#C8102E] font-nfl"
//       style={{ letterSpacing: "0.25em" }}
//     >
//       <div className="flex items-center gap-2">
//         <span className="text-6xl font-bold">Iverson</span>
//       </div>
//       <button className="flex items-center justify-center p-2 hover:bg-opacity-80 rounded">
//         <Menu className="h-12 w-12" />
//       </button>
//     </header>
//   );
// };