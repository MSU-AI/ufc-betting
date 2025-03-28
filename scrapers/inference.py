import requests
from pprint import pprint

resp2 = requests.post("https://generateev-431447820732.us-east1.run.app/inference")
pprint(resp2.json())
