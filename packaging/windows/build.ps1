<#
.SYNOPSIS
  Monta o pacote do Promob Render Studio para Windows 64 bits e, se o Inno Setup
  estiver disponível, gera PromobRenderStudio-Setup.exe e o arquivo .sha256.

.DESCRIPTION
  Entradas fixadas para um build reproduzível:
    * Python: CPython $PythonVersion (python-build-standalone, instalado pelo uv);
    * dependências do núcleo: exportadas de backend/uv.lock com hashes (--require-hashes);
    * interface: frontend/dist, compilada a partir de frontend/package-lock.json (npm ci);
    * uv.exe: a mesma versão usada no build, copiada para instalar o Realismo IA sob demanda.
  Os pesos de IA nunca entram no pacote.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File packaging\windows\build.ps1 -Version 0.2.0
#>
param(
    [string]$Version = "0.2.0",
    [string]$PythonVersion = "3.11.15",
    [string]$OutDir = "build\windows",
    [string]$Iscc = "",
    [switch]$SkipFrontend,
    [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
$Out = Join-Path $Root $OutDir
$Stage = Join-Path $Out "stage"          # vira {app} no computador do usuário
$PyInstall = Join-Path $Out "python-install"

function Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Falhou ($LASTEXITCODE): $Exe $($Arguments -join ' ')" }
}

$Uv = (Get-Command uv -ErrorAction Stop).Source
Step "uv: $(& $Uv --version) ($Uv)"

if (Test-Path $Out) { Remove-Item -Recurse -Force $Out }
New-Item -ItemType Directory -Force $Stage, (Join-Path $Stage "runtime"), (Join-Path $Stage "app") | Out-Null

# 1. Interface compilada --------------------------------------------------------
if (-not $SkipFrontend) {
    Step "Compilando a interface (npm ci + build)"
    Invoke-Checked "npm" @("--prefix", "frontend", "ci")
    Invoke-Checked "npm" @("--prefix", "frontend", "run", "build")
}
if (-not (Test-Path "frontend\dist\index.html")) { throw "frontend\dist\index.html não encontrado." }

# 2. Python privado ---------------------------------------------------------------
Step "Instalando CPython $PythonVersion privado"
Invoke-Checked $Uv @("python", "install", $PythonVersion, "--install-dir", $PyInstall, "--no-bin", "--no-registry")
$PyDir = Get-ChildItem $PyInstall -Directory | Where-Object { $_.Name -like "cpython-$PythonVersion-windows-x86_64*" } | Select-Object -First 1
if (-not $PyDir) { throw "CPython $PythonVersion não encontrado em $PyInstall" }
Copy-Item -Recurse $PyDir.FullName (Join-Path $Stage "python")
$Py = Join-Path $Stage "python\python.exe"
Invoke-Checked $Py @("-c", "import sys; assert sys.maxsize > 2**32; print(sys.version)")

# 3. Dependências ------------------------------------------------------------------
Step "Exportando dependências travadas (núcleo e IA)"
$CoreReq = Join-Path $Out "requirements-core.txt"
$AiReq = Join-Path $Stage "runtime\requirements-ai.txt"
Invoke-Checked $Uv @("export", "--project", "backend", "--locked", "--no-dev", "--no-default-groups", "--no-emit-project", "--format", "requirements-txt", "--output-file", $CoreReq)
Invoke-Checked $Uv @("export", "--project", "backend", "--locked", "--only-group", "ai", "--no-emit-project", "--format", "requirements-txt", "--output-file", $AiReq)

Step "Instalando dependências do núcleo no Python privado"
$env:UV_PYTHON_DOWNLOADS = "never"
Invoke-Checked $Uv @("pip", "install", "--python", $Py, "--break-system-packages", "--require-hashes",
    "--requirement", $CoreReq, "--no-deps", "--link-mode", "copy", "--compile-bytecode", "--no-progress")

# 4. Código do aplicativo --------------------------------------------------------------
Step "Copiando o aplicativo"
$AppDir = Join-Path $Stage "app"
New-Item -ItemType Directory -Force (Join-Path $AppDir "backend"), (Join-Path $AppDir "frontend") | Out-Null
Copy-Item -Recurse "backend\app" (Join-Path $AppDir "backend\app")
Copy-Item -Recurse "backend\scripts" (Join-Path $AppDir "backend\scripts")
Copy-Item "backend\pyproject.toml", "backend\uv.lock" (Join-Path $AppDir "backend")
Copy-Item -Recurse "frontend\dist" (Join-Path $AppDir "frontend\dist")
Copy-Item "packaging\windows\promob_launcher.py" $AppDir
Copy-Item "LICENSE", "README.md" $Stage
Get-ChildItem $AppDir -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
Invoke-Checked $Py @("-m", "compileall", "-q", (Join-Path $AppDir "backend"), (Join-Path $AppDir "promob_launcher.py"))
Set-Content -Path (Join-Path $Stage "VERSION") -Value $Version -Encoding ascii

Step "Copiando uv.exe para instalar o Realismo IA sob demanda"
Copy-Item $Uv (Join-Path $Stage "runtime\uv.exe")

Step "Gerando ícone e lista de licenças de terceiros"
Invoke-Checked $Py @("-s", "packaging\windows\make_assets.py", "--stage", $Stage)

# 5. Teste rápido do pacote: importa o app com o Python privado, sem PATH de desenvolvimento -----
Step "Verificando importação do backend com o Python privado"
$saved = $env:PATH
try {
    $env:PATH = "$env:SystemRoot\System32;$env:SystemRoot"
    Push-Location $AppDir
    Invoke-Checked $Py @("-I", "-c", "import sys; sys.path.insert(0, 'backend'); import app.main, cv2, onnxruntime, numpy; print('ok', cv2.__version__, onnxruntime.__version__)")
    Pop-Location
} finally { $env:PATH = $saved }

$sizeMb = [math]::Round(((Get-ChildItem $Stage -Recurse -File | Measure-Object Length -Sum).Sum / 1MB), 1)
Step "Pacote montado em $Stage ($sizeMb MB)"

# 6. Instalador ------------------------------------------------------------------------
if ($SkipInstaller) { return }
if (-not $Iscc) {
    $candidates = @(
        (Get-Command iscc -ErrorAction SilentlyContinue | ForEach-Object Source),
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
    ) | Where-Object { $_ -and (Test-Path $_) }
    $Iscc = $candidates | Select-Object -First 1
}
if (-not $Iscc) { throw "ISCC.exe (Inno Setup 6) não encontrado. Use -SkipInstaller ou informe -Iscc." }

Step "Compilando o instalador com $Iscc"
$NumericVersion = ($Version -replace '[^0-9.].*$', '').TrimEnd('.')
Invoke-Checked $Iscc @("/Qp", "/DAppVersion=$Version", "/DNumericVersion=$NumericVersion", "/DStageDir=$Stage", "/DOutputDir=$Out", "packaging\windows\installer.iss")
$Setup = Join-Path $Out "PromobRenderStudio-Setup.exe"

if ($env:SIGNTOOL_CERT_BASE64) {
    Step "Assinando o instalador"
    $pfx = Join-Path $env:RUNNER_TEMP "codesign.pfx"
    [IO.File]::WriteAllBytes($pfx, [Convert]::FromBase64String($env:SIGNTOOL_CERT_BASE64))
    try {
        $signtool = Get-ChildItem "${env:ProgramFiles(x86)}\Windows Kits\10\bin\*\x64\signtool.exe" | Sort-Object FullName -Descending | Select-Object -First 1
        Invoke-Checked $signtool.FullName @("sign", "/f", $pfx, "/p", $env:SIGNTOOL_CERT_PASSWORD, "/fd", "sha256",
            "/tr", "http://timestamp.digicert.com", "/td", "sha256", "/d", "Promob Render Studio", $Setup)
    } finally { Remove-Item $pfx -Force -ErrorAction SilentlyContinue }
} else {
    Write-Warning "Instalador NÃO assinado (SIGNTOOL_CERT_BASE64 ausente). O Windows SmartScreen pode exibir um aviso."
}

$hash = (Get-FileHash $Setup -Algorithm SHA256).Hash.ToLower()
"$hash  PromobRenderStudio-Setup.exe" | Set-Content -Path "$Setup.sha256" -Encoding ascii -NoNewline
$setupMb = [math]::Round((Get-Item $Setup).Length / 1MB, 1)
Step "Instalador: $Setup ($setupMb MB)"
Step "SHA-256: $hash"
