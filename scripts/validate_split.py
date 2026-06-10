# -*- coding: utf-8 -*-
"""
Valida o dataset RF-DETR gerado em data/dataset_rfdetr/.

Checa: arquivos em disco x JSON (sem orfaos), sem vazamento de imagem/anotacao
entre splits, bboxes dentro dos limites, placeholder id 0 nunca usado.

Uso: python scripts/validate_split.py
"""
import json, os, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "dataset_rfdetr")
SPLITS = ("train", "valid", "test")


def main():
    all_img_ids = collections.Counter()
    all_ann_ids = collections.Counter()
    all_basenames = collections.Counter()
    problems = []

    for sp in SPLITS:
        sp_dir = os.path.join(OUT_DIR, sp)
        coco = json.load(open(os.path.join(sp_dir, "_annotations.coco.json"), encoding="utf-8"))

        disk = set(f for f in os.listdir(sp_dir) if not f.endswith(".json"))
        json_files = set(im["file_name"] for im in coco["images"])

        faltando = json_files - disk
        if faltando:
            problems.append(f"[{sp}] {len(faltando)} imgs no JSON sem arquivo: {list(faltando)[:5]}")
        orfas = disk - json_files
        if orfas:
            problems.append(f"[{sp}] {len(orfas)} arquivos em disco fora do JSON: {list(orfas)[:5]}")

        cat_ids = set(c["id"] for c in coco["categories"])
        img_by_id = {im["id"]: im for im in coco["images"]}

        for a in coco["annotations"]:
            all_ann_ids[a["id"]] += 1
            if a["image_id"] not in img_by_id:
                problems.append(f"[{sp}] ann {a['id']} image_id inexistente {a['image_id']}")
                continue
            if a["category_id"] not in cat_ids:
                problems.append(f"[{sp}] ann {a['id']} category_id invalido {a['category_id']}")
            im = img_by_id[a["image_id"]]
            x, y, w, h = a["bbox"]
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > im["width"] + 1 or y + h > im["height"] + 1:
                problems.append(f"[{sp}] ann {a['id']} bbox fora dos limites {a['bbox']}")

        for im in coco["images"]:
            all_img_ids[im["id"]] += 1
            all_basenames[im["file_name"]] += 1

        used = set(a["category_id"] for a in coco["annotations"])
        if 0 in used:
            problems.append(f"[{sp}] placeholder id 0 usado por anotacao!")

        print(f"[{sp}] imgs(json)={len(coco['images'])} imgs(disco)={len(disk)} "
              f"anns={len(coco['annotations'])} cats={len(coco['categories'])} classes_usadas={sorted(used)}")

    for label, counter in (("image_id", all_img_ids), ("annotation_id", all_ann_ids), ("basename", all_basenames)):
        dups = {k: v for k, v in counter.items() if v > 1}
        if dups:
            problems.append(f"{label} repetido entre splits: {list(dups)[:10]}")

    print(f"\ntotal imgs: {sum(all_img_ids.values())} | unicas: {len(all_img_ids)}")
    print(f"total anns: {sum(all_ann_ids.values())} | unicas: {len(all_ann_ids)}")

    print("\n=== RESULTADO ===")
    if problems:
        print(f"{len(problems)} PROBLEMA(S):")
        for p in problems:
            print("  -", p)
        raise SystemExit(1)
    print("OK: consistente, sem vazamento, bboxes validos, placeholder intacto.")


if __name__ == "__main__":
    main()
