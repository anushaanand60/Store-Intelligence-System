while ($true) {
    Clear-Host
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "Store Intelligence Monitoring Dashboard"
    Write-Host "Timestamp: $ts"
    Write-Host ""
    Write-Host "Metrics"
    curl.exe -s http://localhost:8000/metrics
    Write-Host ""
    Write-Host "Funnel"
    curl.exe -s http://localhost:8000/funnel
    Write-Host ""
    Write-Host "central_api Recent Logs"
    docker compose logs --tail=5 central_api
    Start-Sleep -Seconds 2
}