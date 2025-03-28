#!/bin/bash
git fetch nba-betting
git subtree pull --prefix=betting-website nba-betting main --squash
chmod +x sync-nba.sh