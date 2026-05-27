"""
TaxiWise — Pre-generate Synthetic Parquet Files
================================================
Run this ONCE locally before deploying to Streamlit Cloud:
    python prepare_data.py

Creates:
    data/raw/yellow_taxi_2023_synthetic.parquet  (~50k rows)
    data/raw/yellow_taxi_2024_synthetic.parquet  (~50k rows)

Then commit data/raw/ to GitHub so Streamlit Cloud reads files directly
instead of generating 200k rows at startup (which causes slow/failed deploys).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils import generate_synthetic_data

RAW_DIR = Path(__file__).parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

N_ROWS = 50_000


def main():
    print("=" * 55)
    print("  TaxiWise — Pre-generating Synthetic Parquet Files")
    print("=" * 55)
    generated = 0
    for year in [2023, 2024]:
        out = RAW_DIR / f"yellow_taxi_{year}_synthetic.parquet"
        if out.exists():
            size_kb = out.stat().st_size // 1024
            print(f"  {out.name} already exists ({size_kb:,} KB) — skipping")
            continue
        print(f"  Generating {N_ROWS:,} rows for {year} …", end=" ", flush=True)
        df = generate_synthetic_data(n_rows=N_ROWS, seed=year, years=[year])
        df.to_parquet(out, index=False, compression="snappy")
        size_kb = out.stat().st_size // 1024
        print(f"done  →  {out.name}  ({size_kb:,} KB)")
        generated += 1
    print("=" * 55)
    if generated:
        print("  Commit these files to GitHub before deploying:")
        print("    git add data/raw/")
        print("    git commit -m 'Add pre-generated synthetic parquet files'")
        print("    git push")
    else:
        print("  All files already exist — nothing to do.")
    print("=" * 55)


if __name__ == "__main__":
    main()
