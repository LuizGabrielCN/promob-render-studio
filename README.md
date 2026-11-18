# Promob Render Studio

Aprimore imagens renderizadas no Promob com um aplicativo local, gratuito e de código aberto. Compare o antes e o depois no navegador, ajuste a iluminação e os materiais e exporte o resultado em PNG.

| Modo | Quando usar |
| --- | --- |
| **Clássico** | Ajuste exposição, balanço de branco, sombras, contraste e nitidez com processamento determinístico. Pode rodar em CPU. |
| **Realismo IA** | Reilumine a cena com modelos locais, incluindo efeitos de spots, fitas de LED e lareira quando detectados. O processamento procura preservar geometria, materiais, objetos e textos. Revise o resultado antes de entregar. |

A super-resolução amplia cada lado em 4×. A resolução final depende da imagem de entrada. Um render de 1024 × 698, por exemplo, resulta em 4096 × 2792.

## Instalação

Use Windows, Python 3.11 e Node.js 24 LTS no fluxo recomendado. Instale [Git](https://git-scm.com/downloads), [uv](https://docs.astral.sh/uv/getting-started/installation/) e [Node.js](https://nodejs.org/download). O Realismo IA foi testado com uma NVIDIA RTX 2060 de 6 GB de VRAM. Ele requer CUDA, driver compatível e espaço para baixar vários gigabytes de modelos. A compatibilidade com outras combinações de sistema e GPU pode variar.

```powershell
git clone https://github.com/LuizGabrielCN/promob-render-studio.git
cd promob-render-studio
uv sync --project backend --python 3.11 --locked
npm --prefix frontend ci
npm --prefix frontend run build
uv run --project backend --locked python run.py
```

Abra [http://127.0.0.1:8000](http://127.0.0.1:8000) se o navegador não abrir automaticamente. Depois da instalação, você também pode usar `start.bat` no Windows.

### Ativar Realismo IA

Execute na raiz do projeto e reinicie o aplicativo após o download.

```powershell
uv run --project backend --locked python backend/scripts/download_models.py --required
uv run --project backend --locked python backend/scripts/download_models.py --verify-only
```

A instalação das dependências e a primeira aquisição dos pesos precisam de internet. O Realismo IA usa os modelos do cache local. A ampliação neural pode baixar o modelo Real-ESRGAN no primeiro uso. Depois que os pesos estiverem disponíveis, o render roda na sua máquina, sem enviar a imagem para um serviço de inferência.

## Primeiro render

1. Exporte uma imagem do Promob em JPG, PNG ou WebP. Você também pode usar **Carregar Amostra Promob** para testar com a cena sintética incluída.
2. Abra a imagem no aplicativo, escolha um preset e ajuste os controles. Para começar com **Realismo IA**, experimente intensidade de **48%** e perfil **Auto**.
3. Clique em **Aprimorar em 4K**, aguarde a conclusão e confira o resultado no comparador. A prévia rápida mostra ajustes fotométricos. A reiluminação neural e a ampliação aparecem no render final.
4. Baixe o PNG e confira detalhes como veios, emendas, objetos e letras pequenas antes de usar a imagem numa apresentação.

## Documentação

- [Guia de uso e solução de problemas](docs/uso.md)
- [Arquitetura, API e modelos](docs/arquitetura.md)
- [Como contribuir e executar testes](CONTRIBUTING.md)

## Licença

O código e a documentação deste projeto usam a [licença MIT](LICENSE). Os pesos e demais dependências mantêm as licenças dos respectivos projetos. Consulte a [relação de modelos](docs/arquitetura.md#modelos-e-licenças).

Projeto independente. Promob é citado para indicar o fluxo de trabalho compatível.
