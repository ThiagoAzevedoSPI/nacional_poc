"""
Gera o dataset RF-DETR a partir do export COCO do CVAT.

Entrada : data/raw/cvat_export_coco.zip  (export bruto do CVAT, formato COCO)
Saida   : data/dataset_rfdetr/{train,valid,test}/  (layout RF-DETR)

Decisoes (ver notes/ e a memoria do projeto):
- Mantem APENAS as imagens anotadas (descarta as sem anotacao).
- Remove a classe vasilhame_bom (0 anotacoes).
- Split estratificado por classe 70/15/15, seed fixa (reproduzivel).
- categories no formato Roboflow: placeholder id 0 + classes reais 1..N.
- Extrai so as imagens necessarias direto do zip (sem descompactar tudo).

Uso: python scripts/build_split.py
"""

import collections
import json
import os
import random
import zipfile

SEED = 42
RATIO_VAL = 0.15
RATIO_TEST = 0.15

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIP_PATH = os.path.join(ROOT, "data", "raw", "cvat_export_coco.zip")
COCO_ENTRY = "annotations/instances_default.json"
OUT_DIR = os.path.join(ROOT, "data", "dataset_rfdetr")
SPLITS = ("train", "valid", "test")

PLACEHOLDER_NAME = "vasilhames"  # categoria id 0 (nunca usada por anotacao)


def basename(fn):
    return fn.replace("\\", "/").split("/")[-1]


def main():
    random.seed(SEED)

    with zipfile.ZipFile(ZIP_PATH) as z:
        d = json.loads(z.read(COCO_ENTRY).decode("utf-8"))

        catname = {c["id"]: c["name"] for c in d["categories"]}

        ann_by_img = collections.defaultdict(list)
        for a in d["annotations"]:
            ann_by_img[a["image_id"]].append(a)

        annotated = [im for im in d["images"] if ann_by_img.get(im["id"])]

        # classe (unica) de cada imagem
        img_class = {}
        for im in annotated:
            cats = set(a["category_id"] for a in ann_by_img[im["id"]])
            assert len(cats) == 1, f"imagem {im['id']} tem multiplas classes: {cats}"
            img_class[im["id"]] = next(iter(cats))

        # split estratificado por classe
        by_class = collections.defaultdict(list)
        for im in annotated:
            by_class[img_class[im["id"]]].append(im)

        assignment = {}
        report = collections.defaultdict(lambda: {"train": 0, "valid": 0, "test": 0})
        for cid in sorted(by_class):
            imgs = sorted(by_class[cid], key=lambda x: x["id"])  # deterministico
            random.shuffle(imgs)
            n = len(imgs)
            n_val = max(1, round(RATIO_VAL * n))
            n_test = max(1, round(RATIO_TEST * n))
            n_train = n - n_val - n_test
            assert n_train > 0, f"classe {catname[cid]} sem treino (n={n})"
            parts = (["train"] * n_train) + (["valid"] * n_val) + (["test"] * n_test)
            for im, sp in zip(imgs, parts, strict=True):
                assignment[im["id"]] = sp
                report[catname[cid]][sp] += 1

        # categories: placeholder id 0 + classes reais usadas
        used_cids = sorted(set(img_class.values()))
        categories = [{"id": 0, "name": PLACEHOLDER_NAME, "supercategory": "none"}]
        for cid in used_cids:
            categories.append({"id": cid, "name": catname[cid], "supercategory": PLACEHOLDER_NAME})

        for sp in SPLITS:
            os.makedirs(os.path.join(OUT_DIR, sp), exist_ok=True)

        # monta imagens/anotacoes por split
        split_images = {sp: [] for sp in SPLITS}
        split_anns = {sp: [] for sp in SPLITS}
        for im in annotated:
            sp = assignment[im["id"]]
            split_images[sp].append(
                {
                    "id": im["id"],
                    "license": im.get("license", 0),
                    "file_name": basename(im["file_name"]),  # flat
                    "height": im["height"],
                    "width": im["width"],
                    "date_captured": im.get("date_captured", 0),
                }
            )
            for a in ann_by_img[im["id"]]:
                split_anns[sp].append(
                    {
                        "id": a["id"],
                        "image_id": a["image_id"],
                        "category_id": a["category_id"],
                        "segmentation": a.get("segmentation", []),
                        "area": a.get("area", a["bbox"][2] * a["bbox"][3]),
                        "bbox": a["bbox"],
                        "iscrowd": a.get("iscrowd", 0),
                    }
                )

        licenses = d.get("licenses", [{"name": "", "id": 0, "url": ""}])
        info = {
            "description": "Vasilhames Nacional - defeitos (RF-DETR)",
            "version": "1.0",
            "year": 2026,
        }

        # mapa basename -> entry (basenames sao unicos; evita problema de acento na pasta)
        entry_by_bn = {}
        for e in z.infolist():
            if not e.is_dir() and basename(e.filename).lower().endswith((".jpg", ".jpeg", ".png")):
                entry_by_bn[basename(e.filename)] = e

        total = 0
        for sp in SPLITS:
            coco = {
                "info": info,
                "licenses": licenses,
                "categories": categories,
                "images": split_images[sp],
                "annotations": split_anns[sp],
            }
            with open(os.path.join(OUT_DIR, sp, "_annotations.coco.json"), "w", encoding="utf-8") as f:
                json.dump(coco, f, ensure_ascii=False)
            for im in split_images[sp]:
                bn = im["file_name"]
                if bn not in entry_by_bn:
                    raise KeyError(f"imagem {bn} nao encontrada no zip")
                with open(os.path.join(OUT_DIR, sp, bn), "wb") as f:
                    f.write(z.read(entry_by_bn[bn]))
                total += 1
            print(f"  {sp}: {len(split_images[sp])} imgs, {len(split_anns[sp])} anns")

    print(f"\ntotal imagens copiadas: {total}")
    print("\n=== distribuicao por classe (train/valid/test) ===")
    for k in sorted(report):
        r = report[k]
        print(f"  {k:28s} {r['train']:3d} / {r['valid']:3d} / {r['test']:3d}")
    tot = {sp: len(split_images[sp]) for sp in SPLITS}
    print(f"\n  TOTAL: train={tot['train']}  valid={tot['valid']}  test={tot['test']}  (={sum(tot.values())})")
    print(f"  categorias ({len(categories)}): {[c['name'] for c in categories]}")
    print(f"  saida: {OUT_DIR}")


if __name__ == "__main__":
    main()
