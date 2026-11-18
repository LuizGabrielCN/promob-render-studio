# Instalador para Windows

Este documento explica como o `PromobRenderStudio-Setup.exe` é montado, o que ele instala e como foi testado. Para usar o app, veja o [README](../README.md).

## Decisões

| Tema | Escolha | Motivo |
| --- | --- | --- |
| Python | CPython 3.11 do projeto [python-build-standalone](https://github.com/astral-sh/python-build-standalone), instalado pelo `uv` durante o build e copiado para `{app}\python` | É uma distribuição completa e relocável, com `pythonw.exe` para abrir o app sem console. Não depende de Python instalado nem altera o PATH ou o registro do sistema. |
| Dependências do núcleo | Exportadas de `backend/uv.lock` com hashes e instaladas no Python privado durante o build (`--require-hashes`) | O modo Clássico, a prévia, o download do PNG e a ampliação 4× (ONNX Runtime) funcionam sem internet depois da instalação. |
| Dependências de IA | Grupo `ai` do `pyproject.toml` (PyTorch CUDA, diffusers, transformers etc.), instalado **sob demanda** pelo `uv.exe` incluído em `{app}\runtime` | Somam cerca de 2,7 GB de download e 5,5 GB em disco e só funcionam com GPU NVIDIA. O instalador fica com cerca de 150 MB. O `uv sync` da instalação por código-fonte continua instalando tudo (`default-groups`). |
| Versões | `runtime\requirements-ai.txt` é exportado do mesmo `uv.lock`, com hashes | As bibliotecas instaladas depois são as mesmas testadas no código-fonte. |
| Pesos | Nunca entram no instalador nem no Git | O Real-ESRGAN vai para `%LOCALAPPDATA%\PromobRenderStudio\models`. Os modelos do Realismo IA usam o cache padrão do Hugging Face, o mesmo de `backend/scripts/download_models.py`. |
| Instalação | Inno Setup 6, por usuário (`PrivilegesRequired=lowest`), em `%LOCALAPPDATA%\Programs\Promob Render Studio` | Não pede administrador. Cria atalhos no menu Iniciar, incluindo **Encerrar**, e aparece em Configurações > Aplicativos. |

## O que o atalho faz

O atalho executa `python\pythonw.exe app\promob_launcher.py`.

1. Uma trava de arquivo em `%LOCALAPPDATA%\PromobRenderStudio` garante uma instância por usuário. Se o app já estiver aberto, o atalho apenas abre o navegador nele.
2. O servidor escuta só em `127.0.0.1`. Se a porta 8000 estiver ocupada, usa a próxima livre (8001 a 8010) ou uma porta escolhida pelo sistema.
3. O navegador abre depois que `/api/health` responde com a identificação do app.
4. Logs ficam em `%LOCALAPPDATA%\PromobRenderStudio\logs`. Se o servidor não iniciar, uma mensagem indica o arquivo de log.
5. Requisições com outro `Host` são recusadas (proteção contra DNS rebinding). As ações de download exigem `application/json`, que outros sites não conseguem enviar sem autorização de CORS.

O atalho **Encerrar Promob Render Studio** e o desinstalador chamam `promob_launcher.py --stop`.

## Ativação de recursos

A API `GET /api/features` informa, para `upscale_4x` e `realismo_ia`, se o recurso está pronto, o tamanho estimado e o progresso. `POST /api/features/<recurso>/activate` inicia ou tenta de novo.

No Realismo IA, a ordem é:

1. `nvidia-smi` confirma GPU NVIDIA, driver 527.41 ou mais novo e pelo menos 4 GB de VRAM (6 GB recomendados). Sem isso, nada é baixado e a interface explica o motivo.
2. As bibliotecas de IA são instaladas no Python privado. O cache do `uv` permite retomar uma falha e é apagado depois do sucesso.
3. Um processo separado importa o PyTorch e confirma `torch.cuda.is_available()`. Se falhar, os modelos não são baixados.
4. Os pesos de `app/core/model_manifest.py` (a mesma lista do script `download_models.py`) são baixados para o cache do Hugging Face. Arquivos já baixados são reaproveitados.

Ao atualizar o app, o instalador substitui o Python privado. As bibliotecas de IA precisam ser ativadas de novo. Os modelos baixados continuam no cache.

## Build

```powershell
powershell -ExecutionPolicy Bypass -File packaging\windows\build.ps1 -Version 0.2.0
```

Requer `uv`, Node.js 24 e Inno Setup 6. Use `-SkipInstaller` para só montar o pacote em `build\windows\stage`. O workflow [`windows-installer.yml`](../.github/workflows/windows-installer.yml) fixa as versões de uv, Python, Node.js e Inno Setup, gera o `.exe` e o `.sha256`, testa a instalação e, em tags `v*`, anexa os arquivos à versão no GitHub Releases.

### Assinatura de código

O build assina o instalador quando os segredos `SIGNTOOL_CERT_BASE64` (certificado `.pfx` em Base64) e `SIGNTOOL_CERT_PASSWORD` existem no repositório. Sem eles, o instalador sai sem assinatura e o build registra um aviso.

## Testes

| Teste | Onde roda | O que cobre |
| --- | --- | --- |
| `tests/unit/test_features.py` | CI e local | Pré-checagem CUDA, falha e nova tentativa, uma ativação por vez, falha de rede sem arquivo parcial |
| `tests/unit/test_launcher.py` | CI e local | Porta ocupada, instância única, identificação do servidor |
| `tests/integration/test_api.py` | CI e local | Amostra pela rota real (`classic`, `upscale_factor=1`), PNG na resolução original, API de recursos |
| `frontend/src/store/useEnhancerStore.test.ts` | CI e local | Botão da amostra, padrões sem modelos, bloqueio antes da ativação, nova tentativa |
| [`scripts/ui_walkthrough.py`](../scripts/ui_walkthrough.py) | CI e local | Interface real em Chrome/Edge headless: amostra, comparação, download do PNG, cartões de ativação, falha de download e nova tentativa. Gera as capturas de [`docs/screenshots`](screenshots) |
| [`packaging/windows/install_test.ps1`](../packaging/windows/install_test.ps1) | CI | Instalação silenciosa como conta comum, PATH sem ferramentas de desenvolvimento, porta 8000 ocupada, abertura repetida, roteiro visual, Encerrar e desinstalação |

O roteiro visual simula a falha de rede apontando `HF_ENDPOINT` para uma porta local fechada. Depois, abre nessa porta um redirecionador para o Hugging Face e clica em **Tentar novamente**, que baixa o modelo real e verifica o SHA-256.
