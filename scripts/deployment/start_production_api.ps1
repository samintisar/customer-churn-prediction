# Start Production API Server (PowerShell)
# Uses ml-conda environment automatically

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting Production API Server" -ForegroundColor Cyan
Write-Host "Environment: ml-conda" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Function to check if conda is available
function Test-CondaAvailable {
    try {
        $null = Get-Command conda -ErrorAction Stop
        return $true
    }
    catch {
        return $false
    }
}

# Check conda availability
if (-not (Test-CondaAvailable)) {
    Write-Host "ERROR: Conda not found in PATH" -ForegroundColor Red
    Write-Host "Please initialize conda first with: conda init powershell" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Activating ml-conda environment..." -ForegroundColor Green

# Start the API using conda run (works without activation)
Write-Host "Starting API server with Waitress (production-ready)..." -ForegroundColor Green
Write-Host ""

# Change to project root directory
Set-Location "$PSScriptRoot\..\.." 

conda run -n ml-conda python scripts/deployment/start_api.py --production --port 5000

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Failed to start API server" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
