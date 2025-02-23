import React from "react";
import { OddsRow } from "./GameDetails";
import {Table, TableHeader, TableRow, TableHead, TableBody, TableCell} from "@/components/ui/table";

type OddsTableProps = {
  oddsData: OddsRow[];
};

const OddsTable: React.FC<OddsTableProps> = ({ oddsData }) => {
  return (
    <div className="mt-6 overflow-x-auto">
      <Table className="w-full text-sm text-gray-700">
        <TableHeader>
          <TableRow>
            <TableHead className="font-normal">Book</TableHead>
            <TableHead className="font-normal">Moneyline</TableHead>
            <TableHead className="font-normal">Win Probability</TableHead>
            <TableHead className="font-normal">Edge</TableHead>
            <TableHead className="font-normal">Expected Value</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {oddsData?.map((row, idx) => (
            <TableRow
              key={idx}
              className="border-b border-gray-200 last:border-0"
            >
              <TableCell className="py-2">{row.book}</TableCell>
              <TableCell className="py-2">{row.moneyline}</TableCell>
              <TableCell className="py-2">{row.probability}</TableCell>
              <TableCell className="py-2">{row.edge}</TableCell>
              <TableCell className="py-2">{row.expectedValue}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
};

export default OddsTable;
