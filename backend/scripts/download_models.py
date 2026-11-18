"""Download explicito dos modelos extras do Realismo IA (nada e baixado implicitamente pelo app).

Uso (a partir da raiz do projeto):
    uv run --project backend python backend/scripts/download_models.py --required
    uv run --project backend python backend/scripts/download_models.py --optional
    uv run --project backend python backend/scripts/download_models.py --verify-only

--required : IDEA-Research/grounding-dino-tiny (~0,69 GB) + facebook/sam2.1-hiera-tiny (~0,16 GB)
             + depth-anything/Depth-Anything-V2-Small-hf (~95 MB)
             + lllyasviel/control_v11f1p_sd15_depth (~689 MB)
             + lllyasviel/ic-light (iclight_sd15_fc.safetensors, ~1,60 GiB)
--optional : lllyasviel/ic-light (iclight_sd15_fbc.safetensors, ~1,60 GiB)
             + lllyasviel/control_v11f1e_sd15_tile (diffusion_pytorch_model.bin, ~1,35 GiB)
             + ByteDance/Hyper-SD (Hyper-SD15-8steps-CFG-lora.safetensors, ~0,27 GB)
--verify-only : Verifica offline no cache local (HF_HOME) sem acessar a rede.

Os arquivos vao para o cache padrao do Hugging Face (HF_HOME), onde o app os le com local_files_only=True.
"""
import argparse
import sys

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


def _merge_jobs(*sources: dict) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for src in sources:
        for repo, patterns in src.items():
            existing = merged.setdefault(repo, [])
            for p in patterns:
                if p not in existing:
                    existing.append(p)
    return merged


def verify_cache(jobs: dict[str, list[str]]) -> int:
    from huggingface_hub import try_to_load_from_cache

    missing = []
    for repo, patterns in jobs.items():
        for filename in patterns:
            cached = try_to_load_from_cache(repo_id=repo, filename=filename)
            if isinstance(cached, str):
                print(f"[ok] {repo}/{filename} -> {cached}", flush=True)
            else:
                print(f"[MISSING] {repo}/{filename}", flush=True)
                missing.append(f"{repo}/{filename}")
    if missing:
        print(f"Total missing: {len(missing)}", file=sys.stderr, flush=True)
        return 1
    print("ALL VERIFIED IN LOCAL CACHE", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--required", action="store_true", help="baixa modelos obrigatorios (DINO, SAM2.1, Depth, ControlNet Depth, IC-Light fc)")
    ap.add_argument("--optional", action="store_true", help="baixa modelos opcionais (IC-Light fbc, ControlNet Tile, Hyper-SD)")
    ap.add_argument("--verify-only", action="store_true", help="verifica cache local offline sem baixar arquivos")
    args = ap.parse_args()

    if args.verify_only:
        sources = []
        if args.required or not args.optional:
            sources.append(REQUIRED)
        if args.optional:
            sources.append(OPTIONAL)
        return verify_cache(_merge_jobs(*sources))

    if not (args.required or args.optional):
        ap.error("informe --required, --optional e/ou --verify-only")

    from huggingface_hub import snapshot_download

    sources = []
    if args.required:
        sources.append(REQUIRED)
    if args.optional:
        sources.append(OPTIONAL)
    jobs = _merge_jobs(*sources)

    for repo, patterns in jobs.items():
        print(f"[download] {repo}: {patterns}", flush=True)
        path = snapshot_download(repo_id=repo, allow_patterns=patterns)
        print(f"[ok] {repo} -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
