#!/bin/bash
# keep Colab sessions active (idle sessions are reclaimed) and pull results periodically
n=0
while true; do
  for v in "$@"; do timeout 60 colab exec -s $v -f /tmp/ping.py > /tmp/ping_$v.log 2>&1; done
  n=$((n+1)); if [ $((n % 5)) -eq 0 ]; then ./scripts/sync.sh "$@" > /dev/null 2>&1; fi
  sleep 100
done
