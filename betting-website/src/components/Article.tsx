// app/articles/[slug]/page.tsx
import Header from "@/components/Header";
import { articleData } from "@/components/ArticleData";
import { notFound } from "next/navigation";

export default function ArticlePage({ params }: { params: { slug: string } }) {
  const article = articleData.find((a) => a.url === params.slug);

  if (!article) return notFound();

  const articleContent: Record<string, string> = {
    "what-is-sports-betting": `
**What is Sports Betting?**

Sports betting is the act of predicting sports results based on statistics and other analytical data. People can wager through a licensed sportsbook or illegally through private brokers.

The term "book" refers to the bookkeeping of wagers, debts, and payouts.
    `,
    "moneyline-bets": `
**Moneyline Bets Explained**

Moneyline bets are straightforward wagers on which team will win.

- A **positive** moneyline (e.g., +200) means you're betting on the underdog.
- A **negative** moneyline (e.g., -150) means you're betting on the favorite.

Our platform uses moneyline statistics to estimate outcomes and payout probabilities.
    `,
    "point-spread-bets": `
**Point Spread Bets**

Point spread betting levels the playing field between teams by assigning a point value:

- A favorite at -6.5 must win by more than 6.5 points.
- An underdog at +6.5 can lose by less than 6.5 or win outright.

Point spread betting is popular in high-scoring sports like basketball and football.
    `,
  };

  return (
    <div>
      <Header />
      <div className="max-w-3xl mx-auto px-6 py-12">
        <h1 className="text-4xl font-bold mb-4">{article.title}</h1>
        <img src={article.image} alt={article.title} className="rounded mb-6" />
        <div className="text-lg text-gray-800 whitespace-pre-line">
          {articleContent[params.slug]}
        </div>
      </div>
    </div>
  );
}
