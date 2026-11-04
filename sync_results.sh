#!/bin/sh
# Sends the results folder to GitHub and fetches any kit updates. Safe to run repeatedly.
set -eu
cd "$(dirname "$0")"
git pull --rebase --autostash --quiet origin main
if [ -d results ]; then
  git add results
  if ! git diff --cached --quiet; then
    git -c user.name="engram-vps" -c user.email="engram-vps@localhost" commit --quiet -m "Results from VPS $(date -u +%Y-%m-%dT%H:%MZ)"
    git push --quiet origin main
    echo "Results pushed."
  else
    echo "No new results."
  fi
fi
