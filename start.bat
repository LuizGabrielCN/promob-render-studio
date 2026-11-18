@echo off
title Promob Render Studio - Aprimorador 4K
chcp 65001 > nul

echo ================================================================================
echo    PROMOB RENDER STUDIO - Inicializador Rapido (Windows)
echo ================================================================================
echo.

where uv >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [INFO] Gerenciador uv detectado. Inicializando ambiente...
    uv run --project backend python run.py %*
    goto :end
)

where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [INFO] Python detectado no sistema. Executando run.py...
    python run.py %*
    goto :end
)

echo [ERRO] Nem 'uv' nem 'python' foram encontrados no PATH do sistema.
echo Por favor, instale o Python 3.11+ ou uv (https://astral.sh/uv) para executar.
pause
exit /b 1

:end
if %ERRORLEVEL% neq 0 (
    echo.
    echo [AVISO] O processo encerrou com codigo de erro %ERRORLEVEL%.
    pause
)
