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
│   └── README.md
│
├── notes/                        # notas do projeto
├── tools/cvat/                   # clone da ferramenta de anotação (fora do pipeline)
├── pyproject.toml                # uv: deps (dev + train), config do ruff e do ty
└── uv.lock                       # versões travadas (inclui torch cu121 p/ máquina de treino)
```

## Ambiente de desenvolvimento (uv + ruff + ty)

O projeto usa [uv](https://docs.astral.sh/uv/) para gerenciar o ambiente (config em `pyproject.toml`,
lock em `uv.lock`). `uv sync` cria a `.venv` com as ferramentas de dev:

```bash
uv sync                  # cria/atualiza a .venv (ruff + ty no grupo dev)
uv run ruff check .      # lint (--fix aplica correções)
uv run ruff format .     # formatação
uv run ty check          # type check (torch/rfdetr geram warning aqui: só existem na máquina de treino)
```

As dependências de treino (torch+CUDA, rfdetr...) ficam no grupo `train` e **não** são instaladas
aqui — só na máquina com GPU, via `uv sync --group train`.

## Como reproduzir o dataset

Requer Python ≥ 3.11 (sem GPU). A partir da raiz do projeto:

```bash
uv run python scripts/analyze_dataset.py     # opcional: inspeciona o export
uv run python scripts/build_split.py         # gera data/dataset_rfdetr/ a partir do zip
uv run python scripts/validate_split.py      # confere integridade
```

Decisões do split: só as **313 imagens anotadas** (as 134 sem anotação foram descartadas),
**70/15/15 estratificado por classe**, seed fixa (42). Detalhes em `notes/` e no README de `training/`.

## Treino

O treino roda em **outra máquina com GPU NVIDIA**. Gere o pacote com
`uv run python scripts/make_training_package.py` (inclui dataset + `training/` + `pyproject.toml`
+ `uv.lock`), copie o zip para lá e siga **`training/README.md`** (`uv sync --group train`).
