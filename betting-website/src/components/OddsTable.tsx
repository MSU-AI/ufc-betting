import React from "react";
import { OddsRow } from "./GameDetails";
import {
  Table,
  TableHeader,
  TableRow,
  TableHead,
  TableBody,
  TableCell,
} from "@/components/ui/table";

type OddsTableProps = {
  oddsData: OddsRow[];
};

const OddsTable: React.FC<OddsTableProps> = ({ oddsData }) => {
  return (
    <div className="mt-6 overflow-x-auto">
      <Table className="w-full text-sm text-gray-700">
        <TableHeader>
          <TableRow>
            <TableHead className="font-bold">Casino</TableHead>
            <TableHead className="font-bold">Moneyline</TableHead>
            <TableHead className="font-bold">Win Probability</TableHead>
            <TableHead className="font-bold">Edge</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {oddsData?.map((row, idx) => {
            // Parse edge value for conditional formatting
            const edgeValue = parseFloat(row.edge);
            const edgeColorClass =
              edgeValue > 0
                ? "text-green-600"
                : edgeValue < 0
                ? "text-red-600"
                : "text-gray-700";
            return (
              <TableRow key={idx} className="border-b border-gray-200 last:border-0">
                <TableCell className="py-2 font-bold">{row.book}</TableCell>
                <TableCell className="py-2 font-bold">{row.moneyline}</TableCell>
                <TableCell className="py-2 font-bold">{row.probability}</TableCell>
                <TableCell className={`py-2 font-bold ${edgeColorClass}`}>
                  {row.edge}
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
};

export default OddsTable;
