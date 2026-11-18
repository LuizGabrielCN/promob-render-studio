# Guia de uso

## Preparar a imagem

Exporte o render do Promob no enquadramento desejado. Use PNG quando precisar conservar detalhes finos da entrada. JPEG e WebP também são aceitos. A ferramenta trabalha sobre a imagem exportada. O projeto, a câmera e as medidas continuam no Promob.

## Ajustar o render

| Controle | Efeito |
| --- | --- |
| **Temperatura e tint** | Ajustam a sensação de luz quente ou fria e a dominante entre verde e magenta. |
| **Exposição, sombras e realces** | Clareiam ou escurecem a imagem e controlam a leitura nas áreas escuras ou muito claras. |
| **CLAHE** | Aumenta o contraste local para destacar relevo e textura. |
| **Oclusão de ambiente** | Reforça sombras de contato em frestas, junções e cantos. |
| **Nitidez e redução de ruído** | Realçam bordas e suavizam granulação. Exagerar pode criar halos ou apagar detalhes. |
| **Brilho GI e reflexos** | Ajusta o realce luminoso em bordas e superfícies. |
| **Acabamento fotográfico** | Aplica uma resposta mais suave nos realces, vinheta leve e grão discreto. |
| **Perfil de cena** | Adapta a reiluminação à cozinha, banheiro, sala ou quarto, área externa ou seleção automática. |
| **Fidelidade Promob ↔ Realismo IA** | Controla a intensidade da reiluminação. Reduza se materiais ou objetos começarem a mudar demais. |
| **Fitas de LED** | Interpreta guias amarelas finas e alongadas da imagem como perfis de luz. |

Em cenas que usam a convenção de guias amarelas, confira o LED na prévia e no resultado. Objetos pequenos, texto de poucos pixels e reflexos intensos merecem revisão ampliada.

## App instalado no Windows

| Sintoma | O que fazer |
| --- | --- |
| Cliquei no atalho e nada abriu. | Aguarde alguns segundos: o navegador abre quando o servidor fica pronto. Se aparecer uma mensagem de erro, ela indica o log em `%LOCALAPPDATA%\PromobRenderStudio\logs`. |
| Abri o app duas vezes. | Não há problema. A segunda abertura só mostra de novo a janela do app que já está rodando. |
| Outro programa usa a porta 8000. | O app escolhe outra porta automaticamente (8001, 8002...). O endereço aparece na barra do navegador. |
| Quero fechar o app. | Use **Encerrar Promob Render Studio** no menu Iniciar. Fechar a aba do navegador não encerra o servidor. |
| O download do modelo 4× ou do Realismo IA falhou. | Confira a conexão e clique em **Tentar novamente**. As partes já baixadas são aproveitadas. |
| O Realismo IA aparece como indisponível. | O recurso precisa de GPU NVIDIA com driver 527.41 ou mais novo e 4 GB de VRAM ou mais (6 GB recomendados). Atualize o driver pelo site da NVIDIA e abra o app de novo. O modo Clássico continua funcionando. |
| Atualizei o app e o Realismo IA pede ativação de novo. | A atualização substitui o Python privado. Ative de novo: as bibliotecas são baixadas outra vez, mas os modelos já baixados são reaproveitados. |

## Solução de problemas

| Sintoma | O que verificar |
| --- | --- |
| A página mostra apenas a API ou a interface não aparece. | Rode `npm --prefix frontend ci` e `npm --prefix frontend run build` na raiz e reinicie. |
| A porta 8000 está em uso (código-fonte). | Rode `uv run --project backend python run.py --port 8001` e abra a URL indicada no terminal. |
| O Realismo IA informa que faltam modelos. | Execute `download_models.py --required` e depois `--verify-only` com os comandos do README. Use o mesmo ambiente e cache ao iniciar o app. |
| Falta memória de vídeo ou o render demora demais. | Feche outros processos que usam a GPU ou selecione o modo clássico. Imagens maiores aumentam o custo de ampliação e a RAM necessárias. |
| O download do modelo 4× falha. | Confira a conexão no primeiro uso e as mensagens do terminal. O peso ONNX é identificado na página de arquitetura. |
| O link de download deixou de funcionar. | Os PNGs ficam num cache com limpeza automática. Salve o arquivo final e, se o cache já foi limpo, renderize novamente. |

O servidor escuta em `127.0.0.1` por padrão. Use o aplicativo na sua máquina. A API foi projetada para uso local e não fornece autenticação de usuários.
