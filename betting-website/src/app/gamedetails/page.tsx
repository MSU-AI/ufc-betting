// src/app/gamedetails/page.tsx
import GameDetailsPage from "@/app/gamedetails/GameDetailsPage";

export default function Page({ searchParams }: { searchParams: { id?: string | string[] } }) {
  return <GameDetailsPage searchParams={searchParams} />;
}
