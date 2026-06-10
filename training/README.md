# Treino — RF-DETR (máquina com GPU)

Esta pasta roda na **máquina mais potente com GPU NVIDIA (CUDA)**, não no PC de anotação.

## 1. O que copiar para a máquina de treino

- `data/dataset_rfdetr/` (o dataset já no formato RF-DETR — ~2 GB)
- esta pasta `training/`

> O resto do projeto (`data/raw/`, `tools/cvat/`, `scripts/`) **não** é necessário para treinar.

## 2. Setup do ambiente

Requer **Python ≥ 3.10** e GPU NVIDIA com driver + CUDA.

```bash
python -m venv .venv
# Linux/Mac: source .venv/bin/activate   |   Windows: .venv\Scripts\activate

# 1) PyTorch COM CUDA (escolha a versão de CUDA da sua máquina):
#    https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2) demais dependências
pip install -r requirements.txt
```

Confirme que a GPU é vista: `python -c "import torch; print(torch.cuda.is_available())"` → `True`.

## 3. Treinar

```bash
# GPU média (T4 / RTX 12–16 GB): batch 4 × acúmulo 4 = batch efetivo 16
python train_rfdetr.py --dataset-dir ../data/dataset_rfdetr --model medium \
    --epochs 50 --batch-size 4 --grad-accum 4 --early-stopping

# GPU grande (A100 / H100):
python train_rfdetr.py --dataset-dir ../data/dataset_rfdetr --model medium \
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

Após treinar, exporte o checkpoint para ONNX:

```python
from rfdetr import RFDETRMedium
model = RFDETRMedium(pretrain_weights="output/checkpoint_best_ema.pth")
model.export()            # gera ONNX em output/
```

O ONNX alimenta TensorRT → DeepStream (a cadeia de deploy das notas originais continua válida).
