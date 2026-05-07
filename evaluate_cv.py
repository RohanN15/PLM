"""5-fold cross-validation evaluation for secondary structure prediction."""

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
import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from torch import nn

from model import ResidueMLPClassifier, get_device
from train_mlp import compute_class_weights, load_residue_tensors, make_loader
from validate_dataset import load_and_validate_dataset


def _evaluate(model, loader, criterion, device):
    """Like train_mlp.evaluate but avoids the PyTorch→NumPy bridge (NumPy 2.x compat)."""
    model.eval()
    total_loss = 0.0
    total_items = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            total_items += y.size(0)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    y_true = np.array(all_labels, dtype=np.int64)
    y_pred = np.array(all_preds, dtype=np.int64)
    accuracy = float((y_true == y_pred).mean())
    return total_loss / total_items, accuracy, y_true, y_pred


def train_one_fold(train_df, test_df, embedding_dir, device, batch_size, epochs, lr, weight_decay, dropout):
    x_train, y_train = load_residue_tensors(train_df, embedding_dir)
    x_test, y_test = load_residue_tensors(test_df, embedding_dir)

    train_loader = make_loader(x_train, y_train, batch_size=batch_size, shuffle=True)
    test_loader = make_loader(x_test, y_test, batch_size=batch_size, shuffle=False)

    model = ResidueMLPClassifier(dropout=dropout).to(device)
    class_weights = compute_class_weights(y_train).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y.size(0)
            n += y.size(0)
        if epoch % 10 == 0:
            print(f"    epoch {epoch:3d}/{epochs}  train_loss={running_loss / n:.4f}")

    _, q3, y_true, y_pred = _evaluate(model, test_loader, criterion, device)
    report = classification_report(
        y_true, y_pred,
        labels=[0, 1, 2],
        target_names=["H", "E", "C"],
        output_dict=True,
        zero_division=0,
    )
    return q3, report


def run_cv(
    df,
    embedding_dir="embeddings",
    output_dir=".",
    n_splits=5,
    random_state=42,
    batch_size=2048,
    epochs=30,
    lr=3e-4,
    weight_decay=1e-4,
    dropout=0.3,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")
    print(f"Running {n_splits}-fold cross-validation on {len(df)} proteins\n")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(df), start=1):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        print(f"=== Fold {fold_idx}/{n_splits}  train={len(train_df)}  test={len(test_df)} ===")

        q3, report = train_one_fold(
            train_df, test_df, embedding_dir, device,
            batch_size, epochs, lr, weight_decay, dropout,
        )
        fold_results.append({
            "fold": fold_idx,
            "q3_accuracy": float(q3),
            "H_f1": float(report["H"]["f1-score"]),
            "E_f1": float(report["E"]["f1-score"]),
            "C_f1": float(report["C"]["f1-score"]),
            "H_precision": float(report["H"]["precision"]),
            "E_precision": float(report["E"]["precision"]),
            "C_precision": float(report["C"]["precision"]),
            "H_recall": float(report["H"]["recall"]),
            "E_recall": float(report["E"]["recall"]),
            "C_recall": float(report["C"]["recall"]),
            "test_proteins": test_df["pdb_id"].tolist(),
            "classification_report": report,
        })
        print(
            f"  Result: Q3={q3:.4f}  "
            f"H_f1={report['H']['f1-score']:.4f}  "
            f"E_f1={report['E']['f1-score']:.4f}  "
            f"C_f1={report['C']['f1-score']:.4f}\n"
        )

    q3_arr = np.array([r["q3_accuracy"] for r in fold_results])
    h_arr = np.array([r["H_f1"] for r in fold_results])
    e_arr = np.array([r["E_f1"] for r in fold_results])
    c_arr = np.array([r["C_f1"] for r in fold_results])

    summary = {
        "n_splits": n_splits,
        "fold_results": fold_results,
        "mean_q3": float(np.mean(q3_arr)),
        "std_q3": float(np.std(q3_arr)),
        "mean_H_f1": float(np.mean(h_arr)),
        "std_H_f1": float(np.std(h_arr)),
        "mean_E_f1": float(np.mean(e_arr)),
        "std_E_f1": float(np.std(e_arr)),
        "mean_C_f1": float(np.mean(c_arr)),
        "std_C_f1": float(np.std(c_arr)),
    }

    out_json = output_dir / "cv_results.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {out_json}")

    _plot_q3_per_fold(fold_results, summary, output_dir)
    _plot_f1_per_fold(fold_results, output_dir)
    _plot_summary_table(summary, output_dir)
    _print_summary(summary)
    return summary


def _plot_q3_per_fold(fold_results, summary, output_dir):
    folds = [r["fold"] for r in fold_results]
    q3_scores = [r["q3_accuracy"] for r in fold_results]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(folds, q3_scores, color="#4e79a7", alpha=0.85, edgecolor="k", linewidth=0.5)
    ax.axhline(summary["mean_q3"], color="crimson", linestyle="--", linewidth=1.5,
               label=f"Mean = {summary['mean_q3']:.4f} ± {summary['std_q3']:.4f}")
    for bar, val in zip(bars, q3_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.002, f"{val:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Q3 Accuracy")
    ax.set_title("Q3 Accuracy per Fold — 5-Fold Cross-Validation")
    ax.set_xticks(folds)
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    path = Path(output_dir) / "cv_q3_per_fold.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def _plot_f1_per_fold(fold_results, output_dir):
    folds = [r["fold"] for r in fold_results]
    h_f1s = [r["H_f1"] for r in fold_results]
    e_f1s = [r["E_f1"] for r in fold_results]
    c_f1s = [r["C_f1"] for r in fold_results]

    x = np.arange(len(folds))
    width = 0.25
    colors = {"H": "#4e79a7", "E": "#f28e2b", "C": "#59a14f"}

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, h_f1s, width, label="H (helix)", color=colors["H"], alpha=0.85, edgecolor="k", linewidth=0.5)
    ax.bar(x,         e_f1s, width, label="E (strand)", color=colors["E"], alpha=0.85, edgecolor="k", linewidth=0.5)
    ax.bar(x + width, c_f1s, width, label="C (coil)",   color=colors["C"], alpha=0.85, edgecolor="k", linewidth=0.5)
    ax.set_xlabel("Fold")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Score per Fold — 5-Fold Cross-Validation")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i}" for i in folds])
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    path = Path(output_dir) / "cv_f1_per_fold.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def _plot_summary_table(summary, output_dir):
    folds = [r["fold"] for r in summary["fold_results"]]
    rows = []
    for r in summary["fold_results"]:
        rows.append([
            f"Fold {r['fold']}",
            f"{r['q3_accuracy']:.4f}",
            f"{r['H_f1']:.4f}",
            f"{r['E_f1']:.4f}",
            f"{r['C_f1']:.4f}",
        ])
    rows.append([
        "Mean ± Std",
        f"{summary['mean_q3']:.4f} ± {summary['std_q3']:.4f}",
        f"{summary['mean_H_f1']:.4f} ± {summary['std_H_f1']:.4f}",
        f"{summary['mean_E_f1']:.4f} ± {summary['std_E_f1']:.4f}",
        f"{summary['mean_C_f1']:.4f} ± {summary['std_C_f1']:.4f}",
    ])

    cols = ["Fold", "Q3 Accuracy", "H F1", "E F1", "C F1"]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)

    # Header styling
    for col_idx in range(len(cols)):
        tbl[(0, col_idx)].set_facecolor("#2c5f8a")
        tbl[(0, col_idx)].set_text_props(color="white", fontweight="bold")

    # Mean row styling
    last = len(rows)
    for col_idx in range(len(cols)):
        tbl[(last, col_idx)].set_facecolor("#dce9f5")
        tbl[(last, col_idx)].set_text_props(fontweight="bold")

    plt.title("5-Fold Cross-Validation Results", pad=12, fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = Path(output_dir) / "cv_results_table.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def _print_summary(summary):
    print("\n" + "=" * 52)
    print("  5-Fold Cross-Validation Summary")
    print("=" * 52)
    print(f"  {'Fold':<10} {'Q3':>8} {'H F1':>8} {'E F1':>8} {'C F1':>8}")
    print("-" * 52)
    for r in summary["fold_results"]:
        print(
            f"  Fold {r['fold']:<5} "
            f"{r['q3_accuracy']:>8.4f} "
            f"{r['H_f1']:>8.4f} "
            f"{r['E_f1']:>8.4f} "
            f"{r['C_f1']:>8.4f}"
        )
    print("-" * 52)
    print(
        f"  {'Mean':<10} "
        f"{summary['mean_q3']:>8.4f} "
        f"{summary['mean_H_f1']:>8.4f} "
        f"{summary['mean_E_f1']:>8.4f} "
        f"{summary['mean_C_f1']:>8.4f}"
    )
    print(
        f"  {'Std':<10} "
        f"{summary['std_q3']:>8.4f} "
        f"{summary['std_H_f1']:>8.4f} "
        f"{summary['std_E_f1']:>8.4f} "
        f"{summary['std_C_f1']:>8.4f}"
    )
    print("=" * 52)


def main():
    parser = argparse.ArgumentParser(description="5-fold CV evaluation for secondary structure prediction.")
    parser.add_argument("--csv", default="dataset.csv")
    parser.add_argument("--embedding-dir", default="embeddings")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--filter-invalid", action="store_true")
    args = parser.parse_args()

    df, _, _ = load_and_validate_dataset(args.csv, filter_invalid=args.filter_invalid)
    run_cv(
        df,
        embedding_dir=args.embedding_dir,
        output_dir=args.output_dir,
        n_splits=args.n_splits,
        random_state=args.random_state,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
    )


if __name__ == "__main__":
    main()
