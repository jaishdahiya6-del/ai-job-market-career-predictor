"""Wrapper: cleans data/raw/jobs_demo.csv (or your real dataset placed
at the same path/columns) into data/processed/jobs_processed.csv.
Run: python scripts/process_dataset.py [path_to_raw_csv]
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.preprocessing.clean import load_and_clean

if __name__ == "__main__":
    raw_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/jobs_demo.csv"
    df = load_and_clean(raw_path)
    out_path = Path("data/processed/jobs_processed.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Processed {len(df)} rows from {raw_path} -> {out_path}")
