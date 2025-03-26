// src/app/articles/page.tsx
import Articles from "@/components/Articles";
import { articleData } from "@/components/ArticleData";

export default function ArticlesPage() {
  return <Articles articles={articleData} />;
}
