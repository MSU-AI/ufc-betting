import { MongoClient } from "mongodb";

const MONGODB_URI = process.env.MONGODB_URI || "";  
const MONGODB_DB = "nba_stats"; 

if (!MONGODB_URI) {
  throw new Error("Please define the MONGODB_URI environment variable");
}

let cachedClient: MongoClient | null = null;

export async function connectToDatabase() {
  if (cachedClient) {
    console.log("Using cached MongoDB connection with database:", cachedClient.db(MONGODB_DB).databaseName);
    return { client: cachedClient, db: cachedClient.db(MONGODB_DB) };
  }

  console.log("Connecting to MongoDB...");
  const client = new MongoClient(MONGODB_URI);
  await client.connect();
  console.log("Connected to MongoDB with database:", client.db(MONGODB_DB).databaseName);

  cachedClient = client;
  return { client, db: client.db(MONGODB_DB) };
}
