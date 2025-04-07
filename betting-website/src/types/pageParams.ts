export interface PageParams {
  params: Promise<{ id: string }>;
}


export interface ArticlePageParams {
  params: Promise<{ slug: string }>;
}
