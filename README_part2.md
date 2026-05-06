# Part 2: Model and Training

This code trains a per-residue MLP classifier on frozen ESM2 embeddings.

Input CSV columns:

```text
pdb_id,sequence,ss_labels,ss3
```

Training uses only `sequence` as input and `ss3` as the H/E/C target. The raw `ss_labels` column is not used for training.

## 1. Validate the Dataset

```bash
python3 validate_dataset.py --csv dataset.csv
```

This checks required fields, sequence/label length matches, and that `ss3` contains only `H`, `E`, or `C`.

## 2. Extract Frozen ESM2 Embeddings

```bash
python3 extract_embeddings.py --csv dataset.csv --embedding-dir embeddings
```

This loads `esm2_t30_150M_UR50D`, extracts layer 30 per-residue embeddings with shape `(L, 640)`, and saves one file per protein:

```text
embeddings/{pdb_id}.pt
```

Existing embeddings are skipped when the cached tensor has the expected shape and matching sequence.

## 3. Train the MLP Classifier

```bash
python3 train_mlp.py \
  --csv dataset.csv \
  --embedding-dir embeddings \
  --epochs 30 \
  --batch-size 2048
```

Outputs:

```text
checkpoints/best_mlp_classifier.pt
checkpoints/final_mlp_classifier.pt
loss_curve.png
val_accuracy_curve.png
confusion_matrix.png
training_summary.json
```

The split is by protein using an 80/20 train/validation split with `random_state=42`.

## Model

Frozen PLM: ESM2 `esm2_t30_150M_UR50D`

Embedding layer: 30

Embedding dimension: 640

Classifier:

```text
Linear(640 -> 256)
ReLU
Dropout(0.3)
Linear(256 -> 128)
ReLU
Dropout(0.3)
Linear(128 -> 3)
```

Learned parameters: 197,379

