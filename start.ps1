# Promob Render Studio — PowerShell Launcher
# Permite iniciar o servidor unificado e interface web com um clique no PowerShell.

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "   PROMOB RENDER STUDIO — Inicializador PowerShell (Windows)" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

$uvCmd = Get-Command uv -ErrorAction SilentlyContinue
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue

if ($uvCmd) {
    Write-Host "[INFO] Gerenciador uv detectado ($($uvCmd.Source)). Inicializando..." -ForegroundColor Green
    & uv run --project backend python run.py @args
} elseif ($pythonCmd) {
    Write-Host "[INFO] Python detectado ($($pythonCmd.Source)). Executando run.py..." -ForegroundColor Green
    & python run.py @args
} else {
    Write-Host "[ERRO] Nem 'uv' nem 'python' foram encontrados no PATH." -ForegroundColor Red
    Write-Host "Instale Python 3.11+ ou uv (https://astral.sh/uv) para iniciar o Promob Render Studio." -ForegroundColor Yellow
    Read-Host "Pressione Enter para sair..."
    exit 1
}
