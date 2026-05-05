import os
import requests
import numpy as np
import pandas as pd
from collections import Counter
from Bio.PDB import MMCIFParser, DSSP
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# ── Step 1: Query RCSB ────────────────────────────────────────────────────────

def query_rcsb(max_rows=200):
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "X-RAY DIFFRACTION",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": 2.5,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_entity_polymer_type",
                        "operator": "exact_match",
                        "value": "Protein",
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": max_rows}},
    }
    r = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json=query)
    r.raise_for_status()
    return [e["identifier"] for e in r.json()["result_set"]]


# ── Step 2: Download PDB + Run DSSP ───────────────────────────────────────────

def download_cif(pdb_id, pdb_dir):
    path = os.path.join(pdb_dir, f"{pdb_id.lower()}.cif")
    if not os.path.exists(path):
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    return path


def process_pdb(pdb_id, parser, pdb_dir):
    try:
        path = download_cif(pdb_id, pdb_dir)
        structure = parser.get_structure(pdb_id, path)
        model = structure[0]
        dssp = DSSP(model, path, dssp='mkdssp', file_type='mmCIF')

        seq, ss = "", ""
        for key in dssp.keys():
            aa = dssp[key][1]
            ss8 = dssp[key][2]
            if aa == "X":  # skip unknown residues
                continue
            seq += aa
            ss += ss8

        if 50 <= len(seq) <= 500:
            return (pdb_id, seq, ss)
    except Exception as e:
        print(f"  Skipping {pdb_id}: {e}")
    return None


# ── Step 3: Map 8-class → 3-class ─────────────────────────────────────────────

_SS_MAP = {"H": "H", "G": "H", "I": "H", "E": "E", "B": "E", "S": "C", "T": "C", "-": "C"}

def map_ss(ss8):
    return "".join(_SS_MAP.get(c, "C") for c in ss8)


# ── Step 5a: PCA of ESM2 embeddings ───────────────────────────────────────────

def plot_pca(df, output_path="pca_diversity.png"):
    try:
        import torch
        import esm

        print("  Loading ESM2 model...")
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model.eval()

        data = [(row["pdb_id"], row["sequence"][:1022]) for _, row in df.iterrows()]

        n_layers = model.num_layers
        avg_embeddings = []
        batch_size = 8
        print(f"  Running ESM2 (layers={n_layers}) on {len(data)} sequences...")
        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch = data[i : i + batch_size]
                try:
                    _, _, tokens = batch_converter(batch)
                    out = model(tokens, repr_layers=[n_layers])
                    reps = out["representations"][n_layers]
                    for j, (_, seq) in enumerate(batch):
                        avg_embeddings.append(reps[j, 1 : len(seq) + 1].mean(0).numpy())
                except Exception as e:
                    print(f"  Batch {i}–{i+len(batch)} failed: {e}")

        if not avg_embeddings:
            print("  No embeddings collected — skipping PCA plot.")
            return

        coords = PCA(n_components=2).fit_transform(np.array(avg_embeddings))

        plt.figure(figsize=(8, 6))
        plt.scatter(coords[:, 0], coords[:, 1], alpha=0.6, edgecolors="k", linewidths=0.3)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Sequence Diversity (PCA of ESM2 Embeddings)")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"  Saved {output_path}")

    except ImportError:
        print("  ESM not installed — skipping PCA plot. Install with: pip install fair-esm")
    except Exception as e:
        print(f"  PCA plot failed: {e}")


# ── Step 5b: Class distribution bar chart ─────────────────────────────────────

def plot_class_distribution(df, output_path="class_distribution.png"):
    counts = Counter("".join(df["ss3"]))
    labels = ["H", "E", "C"]
    values = [counts[l] for l in labels]
    colors = ["#4e79a7", "#f28e2b", "#59a14f"]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=colors)
    plt.xlabel("Secondary Structure Class")
    plt.ylabel("Residue Count")
    plt.title("Class Distribution Across All Sequences")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    pdb_dir = "pdb_files"
    os.makedirs(pdb_dir, exist_ok=True)

    print("Step 1: Querying RCSB (X-ray, resolution ≤ 2.5 Å)...")
    ids = query_rcsb(max_rows=200)
    print(f"  {len(ids)} candidate entries returned")

    print("Step 2: Downloading PDBs and running DSSP...")
    parser = MMCIFParser(QUIET=True)

    records = []
    for pdb_id in ids:
        result = process_pdb(pdb_id, parser, pdb_dir)
        if result:
            records.append(result)
            print(f"  [{len(records)}] {pdb_id} — len {len(result[1])}")
        if len(records) >= 150:
            break

    print(f"  Retained {len(records)} structures after length filter (50–500 AA)")

    print("Step 3 & 4: Mapping labels and saving dataset.csv...")
    df = pd.DataFrame(records, columns=["pdb_id", "sequence", "ss_labels"])
    df["ss3"] = df["ss_labels"].apply(map_ss)
    df.to_csv("dataset.csv", index=False)
    print(f"  Saved dataset.csv ({len(df)} rows)")

    print("Step 5: Generating plots...")
    plot_class_distribution(df)
    plot_pca(df)

    print("Done.")


if __name__ == "__main__":
    main()
