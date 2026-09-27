"""Wrapper script to clean raw dataset into data/processed/jobs_processed.csv.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.preprocessing.clean import load_and_clean
from src.utils.logger import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    raw_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/jobs_demo.csv"
    try:
        df = load_and_clean(raw_path)
        out_path = Path("data/processed/jobs_processed.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info("Successfully processed %d rows from %s -> %s", len(df), raw_path, out_path)
    except Exception as e:
        logger.error("Failed to process dataset: %s", e)
        sys.exit(1)
