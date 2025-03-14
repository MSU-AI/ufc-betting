import { MongoClient } from "mongodb";

const MONGODB_URI = process.env.MONGODB_URI || "";  // will have to store in .env file (.env.local for local development)
const MONGODB_DB = process.env.MONGODB_DB || "bettingDB";

if (!MONGODB_URI) {
  throw new Error("Please define the MONGODB_URI environment variable");
}

let cachedClient: MongoClient | null = null;

export async function connectToDatabase() {
  if (cachedClient) return { client: cachedClient, db: cachedClient.db(MONGODB_DB) };

  const client = new MongoClient(MONGODB_URI);
  await client.connect();
  cachedClient = client;

  return { client, db: client.db(MONGODB_DB) };
}
