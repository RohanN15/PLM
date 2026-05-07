# Protein Secondary Structure Prediction with ESM2

Per-residue secondary structure prediction (Q3: H/E/C) using frozen ESM2 embeddings and a lightweight MLP classifier, evaluated with 5-fold cross-validation.

## Results

| | Q3 Accuracy | H F1 | E F1 | C F1 |
|---|---|---|---|---|
| Single train/val split (80/20) | 0.9340 | 0.9394 | 0.9225 | 0.9373 |
| 5-fold CV mean ± std | **0.9594 ± 0.0098** | 0.9675 ± 0.0083 | 0.9645 ± 0.0106 | 0.9476 ± 0.0118 |

## Pipeline

### Step 1 — Build the Dataset

Downloads PDB structures from RCSB, runs DSSP, maps 8-class → 3-class labels, and saves `dataset.csv`.

```bash
python pipeline.py
```

Output: `dataset.csv` with columns `pdb_id`, `sequence`, `ss_labels`, `ss3`.

Filters: X-ray diffraction, resolution ≤ 2.5 Å, sequence length 50–500 AA. Targets 150 proteins.

To validate the dataset:

```bash
python validate_dataset.py --csv dataset.csv
```

---

### Step 2 — Extract Embeddings

Loads `esm2_t30_150M_UR50D` (frozen), extracts layer-30 per-residue representations with shape `(L, 640)`, and caches one file per protein.

```bash
python extract_embeddings.py --csv dataset.csv --embedding-dir embeddings
```

Output: `embeddings/{pdb_id}.pt` — each file contains keys `pdb_id`, `sequence`, `ss3`, `embedding`.

Existing valid embeddings are skipped automatically.

---

### Step 3 — Train the MLP Classifier

```bash
python train_mlp.py \
  --csv dataset.csv \
  --embedding-dir embeddings \
  --epochs 30 \
  --batch-size 2048
```

Outputs:

```
checkpoints/best_mlp_classifier.pt
checkpoints/final_mlp_classifier.pt
loss_curve.png
val_accuracy_curve.png
confusion_matrix.png
training_summary.json
```

Uses an 80/20 protein-level train/validation split (`random_state=42`).

---

### Step 4 — 5-Fold Cross-Validation

```bash
python evaluate_cv.py --csv dataset.csv --embedding-dir embeddings
```

Trains a fresh model for each fold and evaluates on the held-out proteins. Split is at the protein level to prevent data leakage.

Outputs:

```
cv_results.json
cv_q3_per_fold.png
cv_f1_per_fold.png
cv_results_table.png
```

---

## Model Architecture

| Component | Detail |
|---|---|
| PLM | ESM2 `esm2_t30_150M_UR50D` (frozen) |
| Embedding layer | 30 |
| Embedding dimension | 640 |
| Classifier | Linear(640→256) → ReLU → Dropout(0.3) → Linear(256→128) → ReLU → Dropout(0.3) → Linear(128→3) |
| Learned parameters | 197,379 |
| Loss | Weighted CrossEntropyLoss |
| Optimizer | AdamW (lr=3e-4, weight_decay=1e-4) |
| Epochs | 30 |
| Batch size | 2048 residues |

## Requirements

```bash
pip install -r requirements.txt
```

Dependencies: `pandas`, `numpy`, `torch`, `fair-esm`, `scikit-learn`, `matplotlib`.
