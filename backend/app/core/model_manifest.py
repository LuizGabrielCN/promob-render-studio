"""Arquivos do Hugging Face usados pelo Realismo IA.

Compartilhado entre ``backend/scripts/download_models.py`` (linha de comando) e o
ativador de recursos da interface (``app.core.features``). Os pesos ficam no cache
padrão do Hugging Face (HF_HOME) e nunca são incluídos no Git nem no instalador.
"""

REQUIRED = {
    "IDEA-Research/grounding-dino-tiny": [
        "config.json", "model.safetensors", "preprocessor_config.json",
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "added_tokens.json", "vocab.txt",
    ],
    "facebook/sam2.1-hiera-tiny": [
        "config.json", "model.safetensors", "preprocessor_config.json",
        "processor_config.json", "video_preprocessor_config.json",
    ],
    "depth-anything/Depth-Anything-V2-Small-hf": [
        "config.json", "model.safetensors", "preprocessor_config.json",
    ],
    "lllyasviel/control_v11f1p_sd15_depth": [
        "config.json", "diffusion_pytorch_model.fp16.safetensors",
    ],
    "lllyasviel/control_v11p_sd15_canny": [
        "config.json", "diffusion_pytorch_model.fp16.safetensors",
    ],
    "emilianJR/epiCRealism": [
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder/model.safetensors",
        "tokenizer/merges.txt",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
        "unet/config.json",
        "unet/diffusion_pytorch_model.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    ],
    "lllyasviel/ic-light": [
        "iclight_sd15_fc.safetensors",
    ],
}

OPTIONAL = {
    "lllyasviel/ic-light": ["iclight_sd15_fbc.safetensors"],
    "lllyasviel/control_v11f1e_sd15_tile": ["config.json", "diffusion_pytorch_model.bin"],
    "ByteDance/Hyper-SD": ["Hyper-SD15-8steps-CFG-lora.safetensors"],
}
