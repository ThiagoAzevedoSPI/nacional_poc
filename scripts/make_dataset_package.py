"""Gera um .zip do dataset (data/dataset_rfdetr/) para transferir a maquina com GPU.

O codigo de treino vai via git (clone do repo); so o dataset, que nao e
versionado, precisa ser copiado. O zip preserva o prefixo
`data/dataset_rfdetr/...` — extraia na raiz do clone e o default de
--dataset-dir do training/train_rfdetr.py ja funciona.

Roda da raiz do projeto: python scripts/make_dataset_package.py
"""

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "dataset_rfdetr"
OUT_ZIP = ROOT / "nacional_dataset_rfdetr.zip"


def main() -> int:
    if not DATASET.is_dir():
        print(f"ERRO: nao achei {DATASET}", file=sys.stderr)
        return 1

    # (arquivo_no_disco, nome_dentro_do_zip)
    members = []
    for f in sorted(DATASET.rglob("*")):
        if f.is_dir():
            continue
        arc = Path("data") / "dataset_rfdetr" / f.relative_to(DATASET)
        members.append((f, arc.as_posix()))

    total = len(members)
    total_bytes = sum(f.stat().st_size for f, _ in members)
    print(f"Empacotando {total} arquivos ({total_bytes / 1024 / 1024:,.1f} MB) -> {OUT_ZIP.name}")

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
    print(f"   {zsize / 1024 / 1024:,.1f} MB ({total} arquivos)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
