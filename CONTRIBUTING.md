# Contribuindo

Abra uma issue para discutir bugs ou melhorias. Informe o sistema operacional, a GPU, o modo, os controles usados e os passos para reproduzir. Use imagens sintéticas ou exemplos que você tem autorização para publicar.

## Ambiente de desenvolvimento

Instale as dependências conforme o [README](README.md). Para trabalhar na interface com atualização automática, use dois terminais na raiz do projeto.

```powershell
uv run --project backend --locked python run.py --reload --no-browser
```

```powershell
npm --prefix frontend run dev
```

A interface de desenvolvimento usa o endereço mostrado pelo Vite. O backend executa na porta 8000 por padrão.

## Verificações

```powershell
uv run --project backend --locked python -m pytest -c backend/pyproject.toml tests/unit tests/integration
npm --prefix frontend test
npm --prefix frontend run build
```

A suíte de ponta a ponta fica em `tests/e2e`. Execute-a com `uv run --project backend --locked python -m pytest -c backend/pyproject.toml tests/e2e`. Os casos com GPU exigem CUDA e os modelos no cache local. Os de navegador usam Chrome ou Edge. Alguns cenários visuais dependem de imagens fornecidas localmente por `PROMOB_REFERENCE_PATH`, `CHATGPT_REF_BEFORE_PATH`, `CHATGPT_REF_AFTER_PATH` ou `AMOSTRAS_PROMOB_DIR`. Alguns testes esperam regiões de interesse específicas, definidas no arquivo `tests/e2e/conftest.py`. Esses cenários podem ser ignorados automaticamente quando as referências faltam.

Ao abrir um pull request, descreva o comportamento alterado e a validação que executou. Mantenha pesos, imagens de clientes e resultados de benchmark fora do histórico Git.
