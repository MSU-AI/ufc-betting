import subprocess

def dbcronupdate(event, context):
    # executes job
    result = subprocess.run(['gcloud', 'beta', 'run', 'jobs', 'execute', 'db-cron-jobs-job', '--region', 'us-central1'], capture_output=True, text=True)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
