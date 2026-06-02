#!/usr/bin/env sh
while true; do
  clear
  ts=$(date '+%Y-%m-%d %H:%M:%S')
  printf '%s\n' "Store Intelligence Monitoring Dashboard"
  printf '%s\n' "Timestamp: $ts"
  printf '\n%s\n' "Metrics"
  curl.exe -s http://localhost:8000/metrics
  printf '\n%s\n' "Funnel"
  curl.exe -s http://localhost:8000/funnel
  printf '\n%s\n' "central_api Recent Logs"
  docker compose logs --tail=5 central_api
  sleep 2
done