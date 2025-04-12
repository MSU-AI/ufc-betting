#!/bin/bash

REMOTE_NAME="nba-betting"
REMOTE_URL="https://github.com/DevT02/nba-betting.git"

# Check if remote exists
if ! git remote | grep -q "$REMOTE_NAME"; then
  echo "🔗 Remote '$REMOTE_NAME' not found. Adding it..."
  git remote add $REMOTE_NAME $REMOTE_URL
fi

git fetch $REMOTE_NAME

# Check if working directory is clean
if [ -n "$(git status --porcelain)" ]; then
  echo "❌ Working tree is not clean. Please commit or stash your changes first."
  exit 1
fi

# Check if betting-website folder exists and is tracked by Git
if [ ! -d "betting-website" ] || ! git ls-tree -d HEAD betting-website > /dev/null 2>&1; then
  echo "📁 betting-website not found or not a tracked subtree. Running subtree add..."
  git subtree add --prefix=betting-website $REMOTE_NAME main --squash
else
  echo "🔄 betting-website exists. Running subtree pull..."
  git subtree pull --prefix=betting-website $REMOTE_NAME main --squash
fi
