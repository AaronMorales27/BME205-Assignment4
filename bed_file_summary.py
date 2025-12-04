#!/usr/bin/env python3
"""
bed_file_summary.py

Quickly summarize BED files from the command line:
- File size on disk
- Number of intervals (rows)
- Total bases covered (sum of end - start)
"""

import argparse
import os
import pandas as pd

def summarize_bed(path: str) -> dict:
    """Return basic summary stats for a BED file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    # File size on disk
    size_mb = os.path.getsize(path) / (1024 * 1024)
    
    # Load minimal columns (chrom, start, end)
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        usecols=[0, 1, 2],
        names=["chrom", "start", "end"],
        dtype={"chrom": str, "start": int, "end": int},
    )
    
    num_intervals = len(df)
    total_bp = (df["end"] - df["start"]).sum()
    
    return {
        "file": os.path.basename(path),
        "size_mb": size_mb,
        "intervals": num_intervals,
        "total_bp": total_bp
    }

def main():
    parser = argparse.ArgumentParser(description="Summarize BED file sizes and coverage.")
    parser.add_argument("bed_files", nargs="+", help="One or more BED files to summarize.")
    args = parser.parse_args()

    print(f"{'File':<30} {'Size(MB)':>10} {'Intervals':>15} {'Total_bp':>20}")
    print("-" * 80)

    for bed in args.bed_files:
        try:
            stats = summarize_bed(bed)
            print(f"{stats['file']:<30} {stats['size_mb']:>10.2f} {stats['intervals']:>15,} {stats['total_bp']:>20,}")
        except Exception as e:
            print(f"Error processing {bed}: {e}")

if __name__ == "__main__":
    main()
