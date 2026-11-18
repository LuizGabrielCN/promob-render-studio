<#
.SYNOPSIS
  Teste ponta a ponta do instalador, executado por uma conta comum do Windows.

  1. instala em modo silencioso (por usuário, sem administrador);
  2. ocupa a porta 8000 com outro programa e abre o app pelo comando do atalho;
  3. confere que o servidor escuta só em 127.0.0.1 e usou outra porta;
  4. abre o app de novo e confere que não surge um segundo servidor;
  5. percorre a interface com o Python do app (amostra, comparação, download do PNG,
     falha de download do modelo 4× e nova tentativa) e salva capturas;
  6. encerra pelo atalho "Encerrar", desinstala e confere a remoção.

  O PATH é reduzido ao System32: Python, Git, Node.js e uv do sistema não são usados.
#>
param(
    [Parameter(Mandatory)] [string]$Setup,
    [Parameter(Mandatory)] [string]$Walkthrough,
    [Parameter(Mandatory)] [string]$OutDir
)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force $OutDir | Out-Null
Start-Transcript -Path (Join-Path $OutDir "install-test.log") -Force | Out-Null

function Check($cond, $msg) {
    if (-not $cond) { throw "FALHOU: $msg" }
    Write-Host "OK: $msg"
}

$env:PATH = "$env:SystemRoot\System32;$env:SystemRoot;$env:SystemRoot\System32\WindowsPowerShell\v1.0"
$id = [Security.Principal.WindowsIdentity]::GetCurrent()
$isAdmin = ([Security.Principal.WindowsPrincipal]$id).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
Write-Host "Usuário: $($id.Name) | administrador: $isAdmin"
Check (-not $isAdmin) "teste executado por conta comum (sem privilégios de administrador)"
foreach ($tool in "python", "git", "node", "npm", "uv") {
    Check (-not (Get-Command $tool -ErrorAction SilentlyContinue)) "ferramenta de desenvolvimento '$tool' fora do PATH"
}
$pyLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($pyLauncher) {
    # A imagem do GitHub Actions instala o py.exe em C:Windows. O app não o usa: o atalho
    # chama o pythonw.exe privado por caminho absoluto. Fica registrado para o relatório.
    Write-Host "AVISO: lançador py.exe presente em $($pyLauncher.Source) (imagem do runner; não usado pelo app)"
}

$app = Join-Path $env:LOCALAPPDATA "Programs\Promob Render Studio"
$data = Join-Path $env:LOCALAPPDATA "PromobRenderStudio"
$startMenu = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"

# 1. Instalação ---------------------------------------------------------------
$t = Get-Date
$p = Start-Process $Setup -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/LOG=`"$OutDir\setup.log`"" -Wait -PassThru
Check ($p.ExitCode -eq 0) "instalador terminou com código 0 em $([int]((Get-Date)-$t).TotalSeconds)s"
Check (Test-Path "$app\python\pythonw.exe") "Python privado instalado em $app"
Check (Test-Path "$app\app\frontend\dist\index.html") "interface compilada incluída"
Check (Test-Path "$startMenu\Promob Render Studio.lnk") "atalho no menu Iniciar"
Check (Test-Path "$startMenu\Encerrar Promob Render Studio.lnk") "atalho Encerrar no menu Iniciar"
$uninstall = Get-ChildItem "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall" | Get-ItemProperty | Where-Object DisplayName -eq "Promob Render Studio"
Check ($null -ne $uninstall) "registrado em Aplicativos instalados (HKCU)"
$sizeMb = [math]::Round(((Get-ChildItem $app -Recurse -File | Measure-Object Length -Sum).Sum / 1MB), 0)
Write-Host "Espaço ocupado após instalar: $sizeMb MB"

# Comando real do atalho
$shell = New-Object -ComObject WScript.Shell
$lnk = $shell.CreateShortcut("$startMenu\Promob Render Studio.lnk")
$stopLnk = $shell.CreateShortcut("$startMenu\Encerrar Promob Render Studio.lnk")
Write-Host "Atalho: $($lnk.TargetPath) $($lnk.Arguments)"

# 2. Porta 8000 ocupada por outro programa ----------------------------------------
$blocker = [System.Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, 8000)
$blocker.Start()
try {
    $env:PROMOB_NO_BROWSER = "1"   # CI sem área de trabalho: o navegador é dirigido pelo roteiro visual
    $env:HF_ENDPOINT = "http://127.0.0.1:8765"  # porta fechada = falha de rede real no 1º download
    $t = Get-Date
    Start-Process $lnk.TargetPath -ArgumentList $lnk.Arguments -WorkingDirectory $lnk.WorkingDirectory
    do { Start-Sleep -Milliseconds 500 } until ((Test-Path "$data\instancia.json") -or ((Get-Date) - $t).TotalSeconds -gt 120)
    Check (Test-Path "$data\instancia.json") "servidor pronto em $([int]((Get-Date)-$t).TotalSeconds)s"
    $state = Get-Content "$data\instancia.json" | ConvertFrom-Json
    Check ($state.port -ne 8000) "porta 8000 ocupada: app escolheu a porta $($state.port)"
    $listen = Get-NetTCPConnection -State Listen -OwningProcess $state.pid
    Check (($listen | Where-Object LocalAddress -ne "127.0.0.1").Count -eq 0) "escuta apenas em 127.0.0.1 ($($listen.LocalAddress -join ', '))"
    $health = Invoke-RestMethod "$($state.url)/api/health"
    Check ($health.app -eq "promob-render-studio") "health responde (dispositivo: $($health.gpu))"

    # 3. Abertura repetida ---------------------------------------------------------
    $before = @(Get-Process pythonw -ErrorAction SilentlyContinue).Count
    $p2 = Start-Process $lnk.TargetPath -ArgumentList $lnk.Arguments -WorkingDirectory $lnk.WorkingDirectory -PassThru
    $p2.WaitForExit(60000) | Out-Null
    $after = @(Get-Process pythonw -ErrorAction SilentlyContinue).Count
    Check ($p2.HasExited -and $p2.ExitCode -eq 0) "segunda abertura reutiliza a instância e termina"
    Check ($after -eq $before) "nenhum servidor extra ($after processo(s) pythonw)"
    $state2 = Get-Content "$data\instancia.json" | ConvertFrom-Json
    Check ($state2.instance -eq $state.instance) "mesma instância após abrir de novo"
} finally {
    $blocker.Stop()
}

# 4. Interface: amostra, comparação, PNG, falha e nova tentativa ----------------------
& "$app\python\python.exe" -s $Walkthrough --url $state.url --out (Join-Path $OutDir "screenshots") --failure-proxy-port 8765
Check ($LASTEXITCODE -eq 0) "roteiro visual (amostra, comparação, PNG, falha e nova tentativa do 4×)"
$feat = Invoke-RestMethod "$($state.url)/api/features"
Check ($feat.features.upscale_4x.ready) "modelo 4× pronto após a nova tentativa"
Write-Host "Realismo IA: disponível=$($feat.features.realismo_ia.available) motivo=$($feat.features.realismo_ia.reason)"

# 5. Encerrar e desinstalar ----------------------------------------------------------
Start-Process $stopLnk.TargetPath -ArgumentList $stopLnk.Arguments -Wait
Start-Sleep 2
Check (-not (Get-Process -Id $state.pid -ErrorAction SilentlyContinue)) "atalho Encerrar finalizou o servidor"

$p = Start-Process "$app\unins000.exe" -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART" -Wait -PassThru
Start-Sleep 3  # o desinstalador termina a remoção num processo auxiliar
Check ($p.ExitCode -eq 0) "desinstalador terminou com código 0"
Check (-not (Test-Path $app)) "pasta do programa removida"
Check (-not (Test-Path "$startMenu\Promob Render Studio.lnk")) "atalho removido"
Check (-not (Test-Path "$data\cache")) "cache de imagens removido"
Check (-not (Test-Path "$data\logs")) "logs removidos"
Write-Host "Modelos baixados mantidos (resposta padrão 'Não' no modo silencioso): $(Test-Path "$data\models")"
Write-Host "TODOS OS TESTES PASSARAM"
Stop-Transcript | Out-Null
