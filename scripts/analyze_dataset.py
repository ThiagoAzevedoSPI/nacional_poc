"""
Analise exploratoria do export COCO do CVAT (data/raw/cvat_export_coco.zip).

Mostra: anotadas x sem anotacao por pasta, imagens multi-classe, imagens por
classe, e duplicidade de basename. Nao altera nada.

Uso: python scripts/analyze_dataset.py
"""

import collections
import json
import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIP_PATH = os.path.join(ROOT, "data", "raw", "cvat_export_coco.zip")
COCO_ENTRY = "annotations/instances_default.json"


def folder(fn):
    parts = fn.replace("\\", "/").split("/")
    return parts[-2] if len(parts) >= 2 else "?"


def basename(fn):
    return fn.replace("\\", "/").split("/")[-1]


def main():
    with zipfile.ZipFile(ZIP_PATH) as z:
        d = json.loads(z.read(COCO_ENTRY).decode("utf-8"))

    catname = {c["id"]: c["name"] for c in d["categories"]}
    ann_by_img = collections.defaultdict(list)
    for a in d["annotations"]:
        ann_by_img[a["image_id"]].append(a)

    rows = collections.defaultdict(lambda: {"anotada": 0, "vazia": 0})
    for im in d["images"]:
        key = "anotada" if ann_by_img.get(im["id"]) else "vazia"
        rows[folder(im["file_name"])][key] += 1

    print("=== pasta: anotadas / sem anotacao ===")
    tot_a = tot_v = 0
    for f in sorted(rows):
        a, v = rows[f]["anotada"], rows[f]["vazia"]
        tot_a += a
        tot_v += v
        print(f"  {f:28s} anotadas={a:3d}  vazias={v:3d}")
    print(f"  {'TOTAL':28s} anotadas={tot_a:3d}  vazias={tot_v:3d}\n")

    multi = 0
    for im in d["images"]:
        if len(set(a["category_id"] for a in ann_by_img.get(im["id"], []))) > 1:
            multi += 1
    print("imagens com >1 categoria distinta:", multi, "\n")

    img_per_class = collections.Counter()
    for im in d["images"]:
        for c in set(a["category_id"] for a in ann_by_img.get(im["id"], [])):
            img_per_class[catname[c]] += 1
    print("=== imagens por classe ===")
    for k in sorted(img_per_class):
        print(f"  {k:28s} {img_per_class[k]:3d}")
    print()

    bn = collections.Counter(basename(im["file_name"]) for im in d["images"])
    dups = {k: v for k, v in bn.items() if v > 1}
    print("basenames duplicados entre pastas:", len(dups))
    for k, v in list(dups.items())[:20]:
        print("   ", k, v)


if __name__ == "__main__":
    main()
