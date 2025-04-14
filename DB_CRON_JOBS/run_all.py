import subprocess

subprocess.run(["python", "fetch_h2h_records.py"], check=True)
subprocess.run(["python", "fetch_upcoming_games.py"], check=True)
