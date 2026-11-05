#!/bin/sh
# Sends the results folder to GitHub and fetches any kit updates. Safe to run repeatedly.
# GitHub sometimes rejects this server's key at random, so each network step retries.
set -eu
cd "$(dirname "$0")"
# Pulling can replace a results file while a running test still writes to the old copy, and those answers vanish when it stops.
if pgrep -f "chat_kit.py|vps_kit.py" >/dev/null 2>&1; then
  echo "A test is running, so nothing was synced, to protect its results. Run this again when it finishes."
  exit 0
fi
retry() {
  n=1
  until "$@" 2>/dev/null; do
    if [ "$n" -ge 12 ]; then echo "Gave up after $n tries: $*"; return 1; fi
    n=$((n + 1)); sleep 10
  done
}
retry git pull --rebase --autostash --quiet origin main
if [ -d results ]; then
  git add results
  if ! git diff --cached --quiet; then
    git -c user.name="engram-vps" -c user.email="engram-vps@localhost" commit --quiet -m "Results from VPS $(date -u +%Y-%m-%dT%H:%MZ)"
  fi
  if [ -n "$(git log origin/main..HEAD --oneline 2>/dev/null)" ]; then
    retry git push --quiet origin main
    echo "Results pushed."
  else
    echo "No new results."
  fi
fi
