# Arquitetura

O frontend React e TypeScript envia imagens multipart ao backend FastAPI. O backend usa OpenCV no caminho fotométrico, PyTorch e Diffusers na reiluminação e ONNX Runtime na ampliação. O servidor também serve a interface compilada em `frontend/dist`.

## Processamento

1. A prévia decodifica uma versão reduzida, aplica os ajustes fotométricos e retorna uma imagem rápida para o comparador.
2. O render final executa o modo escolhido. No Realismo IA, a análise de cena estima profundidade e identifica regiões como objetos, textos, telas e áreas emissivas. IC-Light e ControlNet orientam a reiluminação. Máscaras ajudam a recuperar detalhes e cores da referência.
3. O acabamento e os perfis de LED são aplicados à imagem. Real-ESRGAN amplia a imagem por blocos sobrepostos para limitar o uso de memória. Nas áreas de texto protegido, a ampliação usa os traços da imagem reiluminada.
4. O backend grava o PNG no cache local e disponibiliza o download pelo identificador do trabalho.

## API local

A especificação interativa fica em [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) com o servidor em execução.

| Rota | Uso |
| --- | --- |
| `GET /api/health` | Verificar a disponibilidade do servidor. |
| `GET /api/capabilities` | Consultar os recursos do motor e os modelos em cache. |
| `GET /api/presets` | Listar presets e parâmetros. |
| `POST /api/preview` | Enviar uma imagem e receber a prévia. |
| `POST /api/enhance` | Enviar a imagem e receber um `job_id`. |
| `GET /api/jobs/{job_id}` | Acompanhar o estado e o progresso. |
| `GET /api/download/{job_id}` | Baixar o PNG concluído. |

Para experimentar o processamento clássico na resolução original, envie a amostra e consulte o trabalho até o estado `completed`. Substitua `ID` pelo `job_id` retornado.

```powershell
curl.exe -F "file=@frontend/public/samples/promob_sample.jpg" -F "render_mode=classic" -F "upscale_factor=1" http://127.0.0.1:8000/api/enhance
curl.exe http://127.0.0.1:8000/api/jobs/ID
curl.exe -f -o resultado.png http://127.0.0.1:8000/api/download/ID
```

## Modelos e licenças

Os pesos são obtidos dos projetos de origem e ficam fora do repositório. Consulte os termos de cada modelo antes de reutilizar ou redistribuir os pesos.

| Uso | Origem |
| --- | --- |
| Difusão para reiluminação | [epiCRealism](https://huggingface.co/emilianJR/epiCRealism) e [IC-Light](https://huggingface.co/lllyasviel/ic-light), sob CreativeML Open RAIL-M segundo as fichas dos modelos. |
| Condicionamento estrutural | [ControlNet Depth](https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth) e [ControlNet Canny](https://huggingface.co/lllyasviel/control_v11p_sd15_canny). |
| Profundidade e máscaras | [Depth Anything V2 Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf), [Grounding DINO](https://huggingface.co/IDEA-Research/grounding-dino-tiny) e [SAM 2.1](https://huggingface.co/facebook/sam2.1-hiera-tiny). |
| Reconhecimento de texto | [RapidOCR](https://github.com/RapidAI/RapidOCR). |
| Ampliação 4× | [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) e [conversão ONNX usada pelo aplicativo](https://huggingface.co/othytenk/realesrgan-x4plus). |

O script de instalação mostra os arquivos exigidos em cada repositório de modelos. O Real-ESRGAN também pode ser disponibilizado em `models/realesrgan_x4plus.onnx`.
