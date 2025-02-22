import React from "react";

type TeamToggleProps = {
  teamNames: string[];
  selectedTeam: string;
  setSelectedTeam: (team: string) => void;
};

const TeamToggle: React.FC<TeamToggleProps> = ({ teamNames, selectedTeam, setSelectedTeam }) => {
  return (
    <div className="flex justify-center space-x-4">
      {teamNames.map((team) => (
        <button
          key={team}
          onClick={() => setSelectedTeam(team)}
          className={`px-6 py-2 rounded-full transition-colors border
            ${
              selectedTeam === team
                ? "bg-emerald-500 text-white border-emerald-500"
                : "bg-white text-gray-700 border-gray-300 hover:bg-gray-100"
            }`}
        >
          {team}
        </button>
      ))}
    </div>
  );
};

export default TeamToggle;
