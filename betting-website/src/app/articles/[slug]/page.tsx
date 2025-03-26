import Header from "@/components/Header";
import { articleData } from "@/components/ArticleData";
import { notFound } from "next/navigation";

export default function ArticlePage({ params }: { params: { slug: string } }) {
  const article = articleData.find((a) => a.url === params.slug);

  if (!article) return notFound();

  const contentMap: Record<string, string> = {
    "what-is-sports-betting": `
**What is Sports Betting?**

Sports betting is the act of predicting sports results based on statistics and other analytical data. People can wager through a licensed sportsbook or privately.

The term "book" refers to bookkeeping used by oddsmakers to track bets and payouts.
    `,
    "moneyline-bets": `
**Moneyline Bets Explained**

Moneyline bets are placed on who will win. A negative value (e.g. -150) shows the favorite; a positive value (e.g. +200) is the underdog.

This platform models statistics using moneyline data to improve betting accuracy.
    `,
    "point-spread-bets": `
**Point Spread Bets**

Point spreads add/subtract points to equalize the matchup:

- Favorite (-6.5) must win by **more than 6.5**
- Underdog (+6.5) must **lose by less than 6.5** or win

This method is common in football and basketball.
    `,
  };

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900">
      <Header />
      <main className="max-w-3xl mx-auto px-6 py-10">
        <h1 className="text-4xl font-bold mb-6">{article.title}</h1>
        <img src={article.image} alt={article.title} className="w-full rounded-lg mb-6" />
        <div className="text-lg leading-8 whitespace-pre-line">{contentMap[params.slug]}</div>
      </main>
    </div>
  );
}
