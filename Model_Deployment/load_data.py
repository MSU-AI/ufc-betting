from pymongo import MongoClient
import os
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()


def connect_to_mongodb():
    client = MongoClient(os.getenv('MONGODB_URI'))
    print(os.getenv('MONGODB_URI'))
    db = client['nba_stats']
    collection = db['team_averages']
    return collection

def get_fields(collection):
    fields = collection.find_one()
    return fields.keys()

collection = connect_to_mongodb()
fields = get_fields(collection)
pprint(fields)





