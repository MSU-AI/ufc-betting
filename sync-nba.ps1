# PowerShell script to sync nba-betting into ufc-betting/betting-website
$remoteName = "nba-betting"
$remoteUrl = "https://github.com/DevT02/nba-betting.git"
$remoteExists = git remote | Select-String $remoteName
if (-not $remoteExists) {
    Write-Host "Remote '$remoteName' not found. Adding it..."
    git remote add $remoteName $remoteUrl
    git fetch $remoteName
} else {
    Write-Host "Remote '$remoteName' already exists."
    git fetch $remoteName
}
Write-Host "Pulling latest changes from nba-betting into betting-website/"
git subtree pull --prefix=betting-website $remoteName main --squash
