import requests

resp2 = requests.post("https://generateev-431447820732.us-east1.run.app/inference")
print(resp2.json())
