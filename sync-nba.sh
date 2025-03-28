#!/bin/bash

REMOTE_NAME="nba-betting"
REMOTE_URL="https://github.com/DevT02/nba-betting.git"

if ! git remote | grep -q "$REMOTE_NAME"; then
  echo "Remote '$REMOTE_NAME' not found. Adding it..."
  git remote add $REMOTE_NAME $REMOTE_URL
  git fetch $REMOTE_NAME
else
  echo "Remote '$REMOTE_NAME' already exists."
  git fetch $REMOTE_NAME
fi

# Pull latest subtree changes
echo "Pulling latest changes from nba-betting into betting-website/"
git subtree pull --prefix=betting-website $REMOTE_NAME main --squash
