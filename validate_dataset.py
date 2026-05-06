import argparse
from collections import Counter

import pandas as pd


REQUIRED_COLUMNS = ["pdb_id", "sequence", "ss3"]
VALID_LABELS = set("HEC")


def validate_dataframe(df, filter_invalid=False):
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    valid_mask = pd.Series(True, index=df.index)
    issues = []

    non_null_mask = df[REQUIRED_COLUMNS].notna().all(axis=1)
    if not non_null_mask.all():
        bad_count = int((~non_null_mask).sum())
        issues.append(f"{bad_count} rows have null pdb_id, sequence, or ss3")
        valid_mask &= non_null_mask

    length_mask = df.apply(
        lambda row: isinstance(row.get("sequence"), str)
        and isinstance(row.get("ss3"), str)
        and len(row["sequence"]) == len(row["ss3"]),
        axis=1,
    )
    if not length_mask.all():
        bad_count = int((~length_mask).sum())
        issues.append(f"{bad_count} rows have len(sequence) != len(ss3)")
        valid_mask &= length_mask

    label_mask = df["ss3"].apply(
        lambda labels: isinstance(labels, str) and set(labels).issubset(VALID_LABELS)
    )
    if not label_mask.all():
        bad_count = int((~label_mask).sum())
        issues.append(f"{bad_count} rows contain labels outside H/E/C")
        valid_mask &= label_mask

    if issues:
        print("Dataset validation warnings:")
        for issue in issues:
            print(f"  - {issue}")
        if not filter_invalid:
            raise ValueError("Dataset contains invalid rows. Use --filter-invalid to continue.")

    clean_df = df.loc[valid_mask].copy() if filter_invalid else df.copy()
    return clean_df, issues


def summarize_dataset(df):
    counts = Counter("".join(df["ss3"].tolist()))
    total_residues = sum(counts.values())

    print("Dataset summary")
    print(f"  proteins: {len(df)}")
    print(f"  total residues: {total_residues}")
    for label in ["H", "E", "C"]:
        count = counts.get(label, 0)
        pct = 100.0 * count / total_residues if total_residues else 0.0
        print(f"  {label}: {count} ({pct:.2f}%)")

    return {
        "num_proteins": int(len(df)),
        "total_residues": int(total_residues),
        "class_counts": {label: int(counts.get(label, 0)) for label in ["H", "E", "C"]},
        "class_percentages": {
            label: (100.0 * counts.get(label, 0) / total_residues if total_residues else 0.0)
            for label in ["H", "E", "C"]
        },
    }


def load_and_validate_dataset(csv_path, filter_invalid=False):
    df = pd.read_csv(csv_path)
    df, issues = validate_dataframe(df, filter_invalid=filter_invalid)
    summary = summarize_dataset(df)
    return df, summary, issues


def main():
    parser = argparse.ArgumentParser(description="Validate protein secondary structure CSV.")
    parser.add_argument("--csv", default="dataset.csv", help="Input CSV path.")
    parser.add_argument(
        "--filter-invalid",
        action="store_true",
        help="Filter invalid rows after printing warnings.",
    )
    args = parser.parse_args()

    load_and_validate_dataset(args.csv, filter_invalid=args.filter_invalid)


if __name__ == "__main__":
    main()

