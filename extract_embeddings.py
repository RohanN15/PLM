import argparse
import os
from pathlib import Path

CACHE_DIR = Path(".cache").resolve()
CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("TORCH_HOME", str(CACHE_DIR / "torch"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
Path(os.environ["TORCH_HOME"]).mkdir(exist_ok=True)

import torch

from model import ESM_EMBED_DIM, ESM_LAYER, ESM_MODEL_NAME, get_device
from validate_dataset import load_and_validate_dataset


def embedding_path(embedding_dir, pdb_id):
    return Path(embedding_dir) / f"{pdb_id}.pt"


def cached_embedding_is_valid(path, sequence):
    if not path.exists():
        return False
    try:
        item = torch.load(path, map_location="cpu")
    except Exception:
        return False
    embedding = item.get("embedding")
    return (
        isinstance(embedding, torch.Tensor)
        and tuple(embedding.shape) == (len(sequence), ESM_EMBED_DIM)
        and item.get("sequence") == sequence
    )


def load_esm2(device):
    import esm

    model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    model.eval()
    model.to(device)
    return model, alphabet


def extract_embeddings(
    df,
    embedding_dir="embeddings",
    device=None,
    overwrite=False,
):
    os.makedirs(embedding_dir, exist_ok=True)
    device = device or get_device()
    print(f"Using device: {device}")
    print(f"Loading {ESM_MODEL_NAME}...")
    esm_model, alphabet = load_esm2(device)
    batch_converter = alphabet.get_batch_converter()

    saved = 0
    skipped = 0
    with torch.no_grad():
        for _, row in df.iterrows():
            pdb_id = str(row["pdb_id"])
            sequence = str(row["sequence"])
            ss3 = str(row["ss3"])
            out_path = embedding_path(embedding_dir, pdb_id)

            if not overwrite and cached_embedding_is_valid(out_path, sequence):
                skipped += 1
                print(f"Skipping {pdb_id}: cached embedding is valid")
                continue

            _, _, tokens = batch_converter([(pdb_id, sequence)])
            tokens = tokens.to(device)
            output = esm_model(tokens, repr_layers=[ESM_LAYER], return_contacts=False)
            token_representations = output["representations"][ESM_LAYER]
            embedding = token_representations[0, 1 : tokens.shape[1] - 1].detach().cpu()

            if embedding.shape != (len(sequence), ESM_EMBED_DIM):
                raise ValueError(
                    f"{pdb_id}: expected embedding shape "
                    f"({len(sequence)}, {ESM_EMBED_DIM}), got {tuple(embedding.shape)}"
                )

            torch.save(
                {
                    "pdb_id": pdb_id,
                    "sequence": sequence,
                    "ss3": ss3,
                    "embedding": embedding,
                },
                out_path,
            )
            saved += 1
            print(f"Saved {out_path} shape={tuple(embedding.shape)}")

    print(f"Embedding extraction complete: saved={saved}, skipped={skipped}")


def main():
    parser = argparse.ArgumentParser(description="Extract frozen ESM2 per-residue embeddings.")
    parser.add_argument("--csv", default="dataset.csv", help="Input CSV path.")
    parser.add_argument("--embedding-dir", default="embeddings", help="Output embedding directory.")
    parser.add_argument("--filter-invalid", action="store_true", help="Filter invalid CSV rows.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing embeddings.")
    args = parser.parse_args()

    df, _, _ = load_and_validate_dataset(args.csv, filter_invalid=args.filter_invalid)
    extract_embeddings(
        df,
        embedding_dir=args.embedding_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
