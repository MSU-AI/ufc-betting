import requests
import os
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()
token = os.getenv("GCLOUD_TOKEN")

# resp = requests.get("http://127.0.0.1:5000/")
# print(resp.text)

# headers = {"Authorization": f"Bearer {token}"}
# resp = requests.get("https://generateev-431447820732.us-east1.run.app/")
# print(resp.text)

resp2 = requests.post("http://127.0.0.1:8080/inference")
pprint(resp2.json())
