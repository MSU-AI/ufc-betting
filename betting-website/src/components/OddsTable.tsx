import React from "react";
import { OddsRow } from "./GameDetails";

type OddsTableProps = {
  oddsData: OddsRow[];
};

const OddsTable: React.FC<OddsTableProps> = ({ oddsData }) => {
  return (
    <div className="mt-6 overflow-x-auto">
      <table className="w-full border-collapse text-left">
        <thead>
          <tr>
            <th className="py-3 px-4 text-sm font-semibold uppercase text-gray-500">Book</th>
            <th className="py-3 px-4 text-sm font-semibold uppercase text-gray-500">Moneyline</th>
            <th className="py-3 px-4 text-sm font-semibold uppercase text-gray-500">Win Probability</th>
            <th className="py-3 px-4 text-sm font-semibold uppercase text-gray-500">Edge</th>
            <th className="py-3 px-4 text-sm font-semibold uppercase text-gray-500">Expected Value</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-200">
          {oddsData?.map((row, idx) => (
            <tr key={idx} className="hover:bg-gray-50">
              <td className="py-3 px-4">{row.book}</td>
              <td className="py-3 px-4">{row.moneyline}</td>
              <td className="py-3 px-4">{row.probability}</td>
              <td className="py-3 px-4">{row.edge}</td>
              <td className="py-3 px-4">{row.expectedValue}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default OddsTable;
