# Nacional — PoC de detecção de defeitos em vasilhames (RF-DETR)

Detecção de defeitos em botijões de gás (vasilhames P13) por visão computacional.
Pipeline: **fotografar → anotar (CVAT) → exportar COCO → split → treinar RF-DETR → exportar ONNX → DeepStream**.

## Estrutura

```
.
├── data/
│   ├── raw/                      # entradas brutas (NÃO vão para a máquina de treino)
│   │   ├── images/               # fotos originais 24MP (Coleta 01/<categoria>/...)
│   │   └── cvat_export_coco.zip  # export bruto do CVAT (COCO)
│   └── dataset_rfdetr/           # dataset pronto p/ RF-DETR (train/valid/test)  ►► vai p/ treino
│
├── scripts/                      # preparo de dados (roda AQUI, sem GPU)
│   ├── analyze_dataset.py        # análise exploratória do export COCO
│   ├── build_split.py            # gera data/dataset_rfdetr a partir do zip
│   └── validate_split.py         # valida integridade do split
│
├── training/                     # treino (roda na MÁQUINA COM GPU) — ver training/README.md
│   ├── train_rfdetr.py
│   ├── requirements.txt
│   └── README.md
│
├── notes/                        # notas do projeto
└── tools/cvat/                   # clone da ferramenta de anotação (fora do pipeline)
```

## Como reproduzir o dataset

Requer Python ≥ 3.10 (sem GPU). A partir da raiz do projeto:

```bash
python scripts/analyze_dataset.py     # opcional: inspeciona o export
python scripts/build_split.py         # gera data/dataset_rfdetr/ a partir do zip
python scripts/validate_split.py      # confere integridade
```

Decisões do split: só as **313 imagens anotadas** (as 134 sem anotação foram descartadas),
**70/15/15 estratificado por classe**, seed fixa (42). Detalhes em `notes/` e no README de `training/`.

## Treino

O treino roda em **outra máquina com GPU NVIDIA**. Copie `data/dataset_rfdetr/` + `training/`
para lá e siga **`training/README.md`**.
