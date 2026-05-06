import argparse
import json
import os
from pathlib import Path

CACHE_DIR = Path(".cache").resolve()
CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
Path(os.environ["MPLCONFIGDIR"]).mkdir(exist_ok=True)

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from extract_embeddings import embedding_path
from model import (
    ESM_EMBED_DIM,
    ESM_LAYER,
    ESM_MODEL_NAME,
    ID_TO_LABEL,
    LABEL_TO_ID,
    ResidueMLPClassifier,
    count_learned_parameters,
    get_device,
)
from validate_dataset import load_and_validate_dataset


def labels_to_tensor(ss3):
    return torch.tensor([LABEL_TO_ID[label] for label in ss3], dtype=torch.long)


def load_residue_tensors(df, embedding_dir):
    xs = []
    ys = []
    for _, row in df.iterrows():
        pdb_id = str(row["pdb_id"])
        path = embedding_path(embedding_dir, pdb_id)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing embedding for {pdb_id}: {path}. "
                "Run extract_embeddings.py before training."
            )

        item = torch.load(path, map_location="cpu")
        embedding = item["embedding"].float()
        sequence = str(row["sequence"])
        ss3 = str(row["ss3"])

        if tuple(embedding.shape) != (len(sequence), ESM_EMBED_DIM):
            raise ValueError(f"{pdb_id}: bad embedding shape {tuple(embedding.shape)}")
        if item.get("sequence") != sequence or item.get("ss3") != ss3:
            raise ValueError(f"{pdb_id}: cached embedding metadata does not match CSV")

        xs.append(embedding)
        ys.append(labels_to_tensor(ss3))

    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def make_train_val_split(df, val_size=0.2, random_state=42):
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=random_state,
        shuffle=True,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def compute_class_weights(y_train):
    counts = torch.bincount(y_train, minlength=3).float()
    total = counts.sum()
    weights = total / (len(counts) * counts.clamp_min(1.0))
    return weights


def make_loader(x, y, batch_size, shuffle):
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_items = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            total_items += y.size(0)
            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(y.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    accuracy = float((y_true == y_pred).mean())
    return total_loss / total_items, accuracy, y_true, y_pred


def plot_curves(history, output_dir):
    plt.figure(figsize=(7, 5))
    plt.plot(history["epoch"], history["train_loss"], label="Train loss")
    plt.plot(history["epoch"], history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "loss_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(history["epoch"], history["val_accuracy"], label="Validation Q3 accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "val_accuracy_curve.png", dpi=150)
    plt.close()


def plot_confusion(y_true, y_pred, output_dir):
    labels = [0, 1, 2]
    display_labels = [ID_TO_LABEL[i] for i in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Validation Confusion Matrix")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "confusion_matrix.png", dpi=150)
    plt.close()
    return cm


def train_classifier(
    df,
    embedding_dir="embeddings",
    output_dir=".",
    checkpoint_dir="checkpoints",
    val_size=0.2,
    random_state=42,
    batch_size=2048,
    epochs=30,
    lr=3e-4,
    weight_decay=1e-4,
    dropout=0.3,
    device=None,
):
    output_dir = Path(output_dir)
    checkpoint_dir = Path(checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df = make_train_val_split(df, val_size=val_size, random_state=random_state)
    print(f"Train proteins: {len(train_df)}")
    print(f"Validation proteins: {len(val_df)}")

    x_train, y_train = load_residue_tensors(train_df, embedding_dir)
    x_val, y_val = load_residue_tensors(val_df, embedding_dir)
    print(f"X_train shape: {tuple(x_train.shape)}")
    print(f"y_train shape: {tuple(y_train.shape)}")
    print(f"X_val shape: {tuple(x_val.shape)}")
    print(f"y_val shape: {tuple(y_val.shape)}")

    device = device or get_device()
    print(f"Using device: {device}")

    train_loader = make_loader(x_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(x_val, y_val, batch_size=batch_size, shuffle=False)

    model = ResidueMLPClassifier(dropout=dropout).to(device)
    parameter_count = count_learned_parameters(model)
    print(f"Learned parameters: {parameter_count:,}")

    class_weights = compute_class_weights(y_train).to(device)
    print(f"Class weights [H, E, C]: {[round(v, 4) for v in class_weights.cpu().tolist()]}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_accuracy": []}
    best_val_loss = float("inf")
    best_epoch = None
    best_path = checkpoint_dir / "best_mlp_classifier.pt"
    final_path = checkpoint_dir / "final_mlp_classifier.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_items = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y.size(0)
            running_items += y.size(0)

        train_loss = running_loss / running_items
        val_loss, val_accuracy, _, _ = evaluate(model, val_loader, criterion, device)

        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_accuracy"].append(float(val_accuracy))

        print(
            f"Epoch {epoch:03d}/{epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_q3={val_accuracy:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": float(val_loss),
                    "val_accuracy": float(val_accuracy),
                    "config": {
                        "dropout": dropout,
                        "embedding_dim": ESM_EMBED_DIM,
                        "esm_model": ESM_MODEL_NAME,
                        "esm_layer": ESM_LAYER,
                    },
                },
                best_path,
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epochs,
            "config": {
                "dropout": dropout,
                "embedding_dim": ESM_EMBED_DIM,
                "esm_model": ESM_MODEL_NAME,
                "esm_layer": ESM_LAYER,
            },
        },
        final_path,
    )

    final_val_loss, final_val_accuracy, y_true, y_pred = evaluate(
        model, val_loader, criterion, device
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=["H", "E", "C"],
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=["H", "E", "C"],
        zero_division=0,
    )
    cm = plot_confusion(y_true, y_pred, output_dir)
    plot_curves(history, output_dir)

    summary = {
        "plm": ESM_MODEL_NAME,
        "esm_layer": ESM_LAYER,
        "esm_embedding_dim": ESM_EMBED_DIM,
        "esm_frozen": True,
        "classifier_architecture": [
            "Linear(640 -> 256)",
            "ReLU",
            "Dropout(0.3)",
            "Linear(256 -> 128)",
            "ReLU",
            "Dropout(0.3)",
            "Linear(128 -> 3)",
        ],
        "learned_parameter_count": int(parameter_count),
        "loss_function": "weighted CrossEntropyLoss",
        "optimizer": "AdamW",
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_validation_loss": float(best_val_loss),
        "final_validation_loss": float(final_val_loss),
        "final_validation_q3_accuracy": float(final_val_accuracy),
        "class_weights_H_E_C": class_weights.cpu().tolist(),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "history": history,
        "train_proteins": train_df["pdb_id"].tolist(),
        "validation_proteins": val_df["pdb_id"].tolist(),
    }

    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nValidation classification report")
    print(report_text)
    print_report_ready_summary(summary)
    return summary


def print_report_ready_summary(summary):
    f1 = summary["classification_report"]
    print("\nReport-ready summary")
    print(f"PLM used: ESM2 {summary['plm']}")
    print(f"ESM2 embedding dimension: {summary['esm_embedding_dim']}")
    print("ESM2 frozen: yes")
    print(
        "Classifier architecture: Linear(640 -> 256), ReLU, Dropout(0.3), "
        "Linear(256 -> 128), ReLU, Dropout(0.3), Linear(128 -> 3)"
    )
    print("Activation functions: ReLU")
    print("Dropout: 0.3")
    print(f"Learned parameter count: {summary['learned_parameter_count']:,}")
    print("Loss function: weighted cross-entropy")
    print("Optimizer: AdamW")
    print(f"Learning rate: {summary['learning_rate']}")
    print(f"Weight decay: {summary['weight_decay']}")
    print(f"Batch size: {summary['batch_size']}")
    print(f"Number of epochs: {summary['epochs']}")
    print(f"Final validation loss: {summary['final_validation_loss']:.4f}")
    print(f"Final validation Q3 accuracy: {summary['final_validation_q3_accuracy']:.4f}")
    print(
        "Per-class F1 scores: "
        f"H={f1['H']['f1-score']:.4f}, "
        f"E={f1['E']['f1-score']:.4f}, "
        f"C={f1['C']['f1-score']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Train MLP classifier on cached ESM2 embeddings.")
    parser.add_argument("--csv", default="dataset.csv", help="Input CSV path.")
    parser.add_argument("--embedding-dir", default="embeddings", help="Cached embedding directory.")
    parser.add_argument("--output-dir", default=".", help="Directory for plots and summary JSON.")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Checkpoint directory.")
    parser.add_argument("--filter-invalid", action="store_true", help="Filter invalid CSV rows.")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation protein split fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Train/val split seed.")
    parser.add_argument("--batch-size", type=int, default=2048, help="Residue batch size.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=3e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Classifier dropout.")
    args = parser.parse_args()

    df, _, _ = load_and_validate_dataset(args.csv, filter_invalid=args.filter_invalid)
    train_classifier(
        df,
        embedding_dir=args.embedding_dir,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        val_size=args.val_size,
        random_state=args.random_state,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
    )


if __name__ == "__main__":
    main()
