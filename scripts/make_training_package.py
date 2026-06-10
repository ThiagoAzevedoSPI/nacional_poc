"""Gera um .zip do pacote de treino (data/dataset_rfdetr/ + training/) pronto
para transferir para a maquina com GPU.

Preserva a estrutura `data/dataset_rfdetr/...` e `training/...` para que os
comandos documentados em training/README.md (--dataset-dir ../data/dataset_rfdetr)
funcionem apos extrair. Roda da raiz do projeto: python scripts/make_training_package.py
"""
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "dataset_rfdetr"
TRAINING = ROOT / "training"
OUT_ZIP = ROOT / "nacional_rfdetr_training.zip"

# pastas/arquivos a ignorar dentro de training/
SKIP_DIRS = {"output", "__pycache__", ".venv", "venv", ".ipynb_checkpoints"}


def iter_files(base: Path):
    for p in base.rglob("*"):
        if p.is_dir():
            continue
        if any(part in SKIP_DIRS for part in p.relative_to(base).parts):
            continue
        yield p


def main() -> int:
    if not DATASET.is_dir():
        print(f"ERRO: nao achei {DATASET}", file=sys.stderr)
        return 1
    if not TRAINING.is_dir():
        print(f"ERRO: nao achei {TRAINING}", file=sys.stderr)
        return 1

    # (arquivo_no_disco, nome_dentro_do_zip)
    members = []
    for f in iter_files(DATASET):
        arc = Path("data") / "dataset_rfdetr" / f.relative_to(DATASET)
        members.append((f, arc.as_posix()))
    for f in iter_files(TRAINING):
        arc = Path("training") / f.relative_to(TRAINING)
        members.append((f, arc.as_posix()))

    total = len(members)
    total_bytes = sum(f.stat().st_size for f, _ in members)
    print(f"Empacotando {total} arquivos ({total_bytes/1024/1024:,.1f} MB) -> {OUT_ZIP.name}")

    if OUT_ZIP.exists():
        OUT_ZIP.unlink()

    done = 0
    # ZIP_DEFLATED nivel 1: rapido; JPG ja comprimido nao encolhe, JSON sim.
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as zf:
        for fpath, arcname in members:
            zf.write(fpath, arcname)
            done += 1
            if done % 50 == 0 or done == total:
                print(f"  {done}/{total}")

    zsize = OUT_ZIP.stat().st_size
    print(f"OK: {OUT_ZIP}")
    print(f"   {zsize/1024/1024:,.1f} MB ({total} arquivos)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
