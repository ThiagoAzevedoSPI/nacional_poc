"""
Treino do RF-DETR no dataset de defeitos de vasilhames.

RODA NA MAQUINA COM GPU CUDA (nao roda no PC de anotacao).

Dataset esperado (formato RF-DETR / COCO):
    <dataset-dir>/
        train/  _annotations.coco.json + imagens
        valid/  _annotations.coco.json + imagens
        test/   _annotations.coco.json + imagens

Exemplos:
    # GPU media (T4/RTX, ~12-16GB): batch 4 x acumulo 4 = batch efetivo 16
    python train_rfdetr.py --dataset-dir ../data/dataset_rfdetr --model medium \
        --epochs 50 --batch-size 4 --grad-accum 4

    # GPU grande (A100/H100): batch 16 x acumulo 1
    python train_rfdetr.py --dataset-dir ../data/dataset_rfdetr --model medium \
        --epochs 50 --batch-size 16 --grad-accum 1

Mantenha batch-size * grad-accum = 16 (batch efetivo recomendado pelo RF-DETR).
"""

import argparse
import os
import sys

MODELS = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}


def parse_args():
    p = argparse.ArgumentParser(description="Treino RF-DETR (defeitos de vasilhames)")
    here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument(
        "--dataset-dir",
        default=os.path.join(here, "..", "data", "dataset_rfdetr"),
        help="pasta com train/ valid/ test/ (default: ../data/dataset_rfdetr)",
    )
    p.add_argument(
        "--model",
        choices=list(MODELS),
        default="medium",
        help="tamanho do modelo (default: medium)",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4, help="batch por GPU")
    p.add_argument("--grad-accum", type=int, default=4, help="passos de acumulo de gradiente")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="resolucao quadrada (DEVE ser divisivel por 56). Default: a do modelo.",
    )
    p.add_argument("--output-dir", default=os.path.join(here, "output"))
    p.add_argument("--early-stopping", action="store_true", help="para cedo se nao melhorar")
    p.add_argument("--tensorboard", action="store_true")
    p.add_argument("--wandb", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.resolution is not None and args.resolution % 56 != 0:
        sys.exit(f"--resolution deve ser divisivel por 56 (recebi {args.resolution})")

    # checagens de ambiente
    try:
        import torch
    except ImportError:
        sys.exit("PyTorch nao instalado. Na raiz do pacote, rode: uv sync --group train (ver training/README.md).")

    if not torch.cuda.is_available():
        print(
            "AVISO: CUDA NAO disponivel. O treino do RF-DETR e' inviavel sem GPU. "
            "Confirme drivers NVIDIA + torch com CUDA.",
            file=sys.stderr,
        )
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda} | torch {torch.__version__}")

    ds = os.path.abspath(args.dataset_dir)
    for split in ("train", "valid", "test"):
        j = os.path.join(ds, split, "_annotations.coco.json")
        if not os.path.isfile(j):
            sys.exit(f"nao encontrei {j}. Confira --dataset-dir.")

    from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall  # noqa: F401

    ModelClass = {v: globals()[v] for v in MODELS.values()}[MODELS[args.model]]

    print(
        f"modelo={args.model} epochs={args.epochs} batch={args.batch_size} "
        f"grad_accum={args.grad_accum} (efetivo={args.batch_size * args.grad_accum}) lr={args.lr}"
    )
    print(f"dataset={ds}\noutput={os.path.abspath(args.output_dir)}")

    model_kwargs = {}
    if args.resolution is not None:
        model_kwargs["resolution"] = args.resolution
    model = ModelClass(**model_kwargs)

    train_kwargs = dict(
        dataset_dir=ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        output_dir=os.path.abspath(args.output_dir),
    )
    if args.early_stopping:
        train_kwargs["early_stopping"] = True
    if args.tensorboard:
        train_kwargs["tensorboard"] = True
    if args.wandb:
        train_kwargs["wandb"] = True

    model.train(**train_kwargs)

    print("\nTreino concluido. Checkpoints em:", os.path.abspath(args.output_dir))
    print("Para exportar ONNX (deploy DeepStream/TensorRT):")
    print("  model.export(output_dir='output/onnx')   # ver training/README.md")


if __name__ == "__main__":
    main()
