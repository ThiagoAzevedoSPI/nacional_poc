# Treino — RF-DETR (máquina com GPU)

Esta pasta roda na **máquina mais potente com GPU NVIDIA (CUDA)**, não no PC de anotação.

## 1. O que levar para a máquina de treino

O código vai via git; só o dataset (não versionado, ~2 GB) precisa ser copiado:

```bash
# na máquina de treino:
git clone https://github.com/ThiagoAzevedoSPI/nacional.git
```

No PC de preparo, gere o zip do dataset com `uv run python scripts/make_dataset_package.py`
(`nacional_dataset_rfdetr.zip`), copie-o para a máquina de treino e **extraia na raiz do clone** —
ele recria `data/dataset_rfdetr/` no lugar certo.

## 2. Setup do ambiente (uv)

Requer GPU NVIDIA com driver + CUDA e o [uv](https://docs.astral.sh/uv/) instalado:

```bash
# instalar uv (se ainda não tiver):
curl -LsSf https://astral.sh/uv/install.sh | sh

# na raiz do clone:
uv sync --group train
```

O `uv sync` baixa o Python necessário sozinho (se preciso) e instala **torch 2.5.1+cu121** (índice
oficial do PyTorch, fixado no `pyproject.toml`), `rfdetr` e as demais dependências travadas no
`uv.lock` — além de ruff e ty (grupo dev). Não precisa instalar torch separado nem ativar a venv:
rode tudo com `uv run`.

> Se o driver da máquina exigir outra versão de CUDA, troque o índice `pytorch-cu121` no
> `pyproject.toml` (ex.: `.../whl/cu124`) e rode `uv lock` de novo.

Confirme que a GPU é vista: `uv run python -c "import torch; print(torch.cuda.is_available())"` → `True`.

## 3. Treinar

Da raiz do clone (o `--dataset-dir` default já aponta para `data/dataset_rfdetr/`):

```bash
# GPU média (T4 / RTX 12–16 GB): batch 4 × acúmulo 4 = batch efetivo 16
uv run python training/train_rfdetr.py --model medium \
    --epochs 50 --batch-size 4 --grad-accum 4 --early-stopping

# GPU grande (A100 / H100):
uv run python training/train_rfdetr.py --model medium \
    --epochs 50 --batch-size 16 --grad-accum 1 --early-stopping
```

Regra prática do RF-DETR: manter **`batch-size × grad-accum = 16`** (batch efetivo).
Modelos: `nano | small | medium | large`. `--resolution` (se usar) **deve ser divisível por 56**.

Checkpoints e logs vão para `training/output/`.

## 4. Dataset (resumo)

- 313 imagens anotadas, split estratificado **70/15/15** (train 219 / valid 47 / test 47).
- **9 classes** de defeito (ids 1–9) + categoria placeholder `vasilhames` (id 0, não usada) —
  formato Roboflow exigido pelo RF-DETR.
- Classes: `alca_quebrado_amassado, enferrujado, outra_marca, sem_borracha, sem_tara,
  sujo, tara_com_defeito, tara_sobre_tara, dupla_tara`.

> Observação: as imagens são 24 MP (6000×4000). O RF-DETR redimensiona internamente,
> mas se o data loading ficar lento, pré-reduza as imagens (escalando os bboxes junto).

## 5. Export para deploy (DeepStream / TensorRT)

Após treinar, exporte o checkpoint para ONNX (`uv run python` da raiz do clone):

```python
from rfdetr import RFDETRMedium
model = RFDETRMedium(pretrain_weights="training/output/checkpoint_best_ema.pth")
model.export()            # gera ONNX em output/
```

O ONNX alimenta TensorRT → DeepStream (a cadeia de deploy das notas originais continua válida).
