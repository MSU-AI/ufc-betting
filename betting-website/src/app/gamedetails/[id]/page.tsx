import GameDetailsPage from "@/app/gamedetails/GameDetailsPage";
import { PageParams } from "@/types/pageParams";

export default async function Page({ params }: PageParams) {
  const { id } = await params;
  return <GameDetailsPage id={id} />;
}
