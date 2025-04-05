import google.auth
import google.auth.transport.requests
import requests

def dbcronupdate(event, context):
    project = "civil-cascade-454901-m2"  
    region = "us-central1"
    job_name = "db-cron-jobs-job"
    
    url = f"https://{region}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/{project}/jobs/{job_name}:run"
    
    credentials, _ = google.auth.default()
    auth_req = google.auth.transport.requests.Request()
    credentials.refresh(auth_req)
    token = credentials.token

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(url, headers=headers, json={})
    print("Response Code:", response.status_code)
    print("Response Text:", response.text)
