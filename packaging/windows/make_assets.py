"""Gera o ícone do app e a lista de licenças de terceiros dentro do pacote montado.

Executado pelo build.ps1 com o Python privado (que já contém o Pillow).
"""
import argparse
import sys
from importlib import metadata
from pathlib import Path

from PIL import Image, ImageDraw


def make_icon(path: Path) -> None:
    """Desenha o ícone a partir do favicon.svg (fundo grafite, montanha e brilho)."""
    size = 256
    s = size / 32.0
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=int(8 * s), fill="#18181b", outline="#27272a", width=int(2 * s))
    mountain = [(8, 22), (14, 10), (20, 18), (24, 13), (26, 22)]
    d.polygon([(x * s, y * s) for x, y in mountain], fill="#3f3f46", outline="#a1a1aa", width=int(1.5 * s))
    r = 2.5 * s
    d.ellipse([11 * s - r, 9 * s - r, 11 * s + r, 9 * s + r], fill="#f43f5e")
    star = [(22, 6), (23, 9), (26, 10), (23, 11), (22, 14), (21, 11), (18, 10), (21, 9)]
    d.polygon([(x * s, y * s) for x, y in star], fill="#e4e4e7")
    img.save(path, format="ICO", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])


def write_third_party(path: Path) -> None:
    lines = [
        "Promob Render Studio — componentes de terceiros incluídos no instalador",
        "=" * 72,
        "",
        f"Python {sys.version.split()[0]} (python-build-standalone) — PSF License. Veja python/LICENSE.txt.",
        "uv (runtime/uv.exe) — Apache-2.0 OR MIT, https://github.com/astral-sh/uv",
        "",
        "Pacotes Python (a licença completa de cada um está na respectiva pasta *.dist-info):",
        "",
    ]
    rows = []
    for dist in metadata.distributions():
        name = dist.metadata.get("Name") or "?"
        version = dist.version
        lic = (dist.metadata.get("License-Expression") or dist.metadata.get("License") or "").strip()
        if not lic or len(lic) > 80:
            classifiers = [c.split("::")[-1].strip() for c in dist.metadata.get_all("Classifier") or [] if c.startswith("License ::")]
            lic = ", ".join(classifiers) or (lic.splitlines()[0][:80] if lic else "ver dist-info")
        rows.append(f"{name} {version} — {lic}")
    lines += sorted(set(rows), key=str.lower)
    lines += [
        "",
        "Os pesos de IA (Real-ESRGAN, Stable Diffusion/epiCRealism, ControlNet, IC-Light, Grounding DINO,",
        "SAM 2.1, Depth Anything V2) NÃO fazem parte do instalador. Eles são baixados sob demanda e",
        "mantêm as licenças dos respectivos projetos (ver docs/arquitetura.md no repositório).",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    args = ap.parse_args()
    stage = Path(args.stage)
    make_icon(stage / "app" / "promob-render-studio.ico")
    write_third_party(stage / "THIRD_PARTY_LICENSES.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
