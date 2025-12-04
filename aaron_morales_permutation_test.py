import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd

def read_bed_as_dataframe(path: str) -> pd.DataFrame: # input path string return pd df
    """
    Read a BED-like file (3-column: chrom, start, end) into a pandas DataFrame.

    Purpose:
        Provides a structured representation of genomic intervals for efficient computation.

    How it fits the pseudocode:
        This implements the "load SetA" and "load SetB" steps before building per-chromosome indexes.

    Explanation:
        - Uses pandas for fast I/O (handles .bed or .bed.gz automatically).
        - Filters out malformed or zero-length intervals.
        - Keeps all overlapping intervals (no merging).

    Returns:
        DataFrame with columns ['chrom', 'start', 'end']
    """    

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(
        path,
        sep=r"\s+",              # whitespace/tab separated
        header=None,
        comment="#",
        usecols=[0, 1, 2],       # Utilize the 3 headers 
        compression="infer",
        names=["chrom", "start", "end"],
        dtype={"chrom": str, "start": np.int64, "end": np.int64}
    )

    # Filter invalid entries (end <= start)         IFF messy data
    valid = (df["start"] >= 0) & (df["end"] > df["start"])
    if not valid.all():
        dropped = (~valid).sum()
        sys.stderr.write(f"Warning: Dropping {dropped} invalid/zero-length intervals in {os.path.basename(path)}\n")
        df = df[valid]
    

    return df.reset_index(drop=True)

def build_setB_index(dfB: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Build an efficient lookup index for SetB intervals, grouped by chromosome.

    Purpose:
        Enables fast overlap search via binary search (np.searchsorted).

    How it fits the pseudocode:
        This corresponds to the preprocessing step:
            "for each chromosome in SetB, sort intervals by start position and store starts/ends arrays"

    Explanation:
        - Sorting by start allows us to use binary search later.
        - Each chromosome stores two numpy arrays:
              starts[chrom] = [b1_start, b2_start, ...]
              ends[chrom]   = [b1_end, b2_end, ...]
        - Access pattern is cache-friendly and avoids nested loops over all B intervals.

    Returns:
        dict { chrom: (starts_array, ends_array) }
    """
    index = {}
    
    for chrom, group in dfB.groupby("chrom", sort=False): # Leverages groupby, splits up and iterates df by label name and filtered subframe
        grp_sorted = group.sort_values("start", kind="mergesort") # sorts each subgroup by start position(mergesort is good)
        starts = grp_sorted["start"].to_numpy(dtype=np.int64) # Convert subgroup start & end columns into NumPy arrays
        ends = grp_sorted["end"].to_numpy(dtype=np.int64) # NumPy arrays can perform vector operations and binary search! (efficient)
        index[chrom] = (starts, ends) # dictionary with e.g chr1 : NumPy arrays of start and ends for that chr

    return index # only start values are sorted monotonically, but the ends array share the same index of their corresponding starts

def compute_total_overlap(
    setA_df: pd.DataFrame,
    # setB_df: pd.DataFrame,
    setB_index: Dict[str, Tuple[np.ndarray, np.ndarray]],
    chroms_include: Optional[set] = None
) -> int:
    """
    Compute total base-pair overlap between SetA and SetB.

    Purpose:
        Implements the assignment pseudocode using binary search to limit comparisons.

    Core logic (per pseudocode):
        for each region_A in SetA:
            if region_A.chrom not in SetB: continue
            i = searchsorted(B_starts, A.start)  # first B starting >= A.start
            j  = searchsorted(B_starts, A.end)    # first B starting >= A.end
            candidate_window = B[i0-1 : j]        # possible overlaps
            compute overlap = max(0, min(A.end, B.end) - max(A.start, B.start))
            sum overlaps

    Explanation of binary search here:
        - `np.searchsorted` performs a binary search over the sorted starts array.
        - This narrows the search from all N_B intervals to only those with start positions
          near A.start and A.end (O(log N_B) per query).
        - We still check small local subsets explicitly, as B intervals may overlap slightly beyond these bounds.

    Returns:
        Total overlap in base pairs (integer).
    """
    
    total = np.int64(0)
    # setB_index = build_setB_index(setB_df) # create b_index dictionary

    # Group A intervals by chromosome for efficiency
    for chrom, groupA in setA_df.groupby("chrom", sort=False): # Iterate by chrom and setA sub df
        if chroms_include and chrom not in chroms_include: # chrom is not in definition parameter(don't care)
            continue
        if chrom not in setB_index: # chrom in setA_df not present in the setB_index, skip
            continue

        B_starts, B_ends = setB_index[chrom] # Pull out b start, b stop sorted arrays for particular chrom

        for a_start, a_end in zip(groupA["start"].values, groupA["end"].values): # iterate setA sub df start and stop col.vals(list)
            # Binary search step 1: find insertion point of a_start
            i = np.searchsorted(B_starts, a_start, side="left")
            # Binary search step 2: find insertion point of a_end
            j = np.searchsorted(B_starts, a_end, side="left") # setB_index not sorted by ends, a_end index is somewhere less than a b_start of similiar value

            # Expand search by one interval to catch overlapping regions before a_start
            cand_lo = max(0, i - 1)
            cand_hi = j  # exclusive

            # Candidate subset of sorted B intervals
            b_s = B_starts[cand_lo:cand_hi] # lower - upper region, i - 1 to include end case overlap
            b_e = B_ends[cand_lo:cand_hi] # b_starts is sorted, but b_ends has matching indexes for ends

            # Vectorized overlap calculation (per pseudocode)
            left = np.maximum(a_start, b_s) # a_start is a single value but compares like to lists [a_start, a_start, ...] to [b_s]
            right = np.minimum(a_end, b_e) # selects largest and least value from these two comparisons to find the true interval
            overlap_bp = np.clip(right - left, a_min=0, a_max=None) # case where overlap is negative(a_end < b_start), sets to zero

            total += overlap_bp.sum()

    return int(total)

def load_fai(path: str) -> dict:
    """Load genome FASTA index (.fai) as dictionary {chrom: length}."""
    try:
        fai_df = pd.read_csv(
            path,
            sep="\t",   
            header=None,
            usecols=[0, 1],
            names=["chrom", "length"],
            dtype={"chrom": str, "length": np.int64},
        )
    except Exception as e:
        sys.exit(f"Error loading genome index {path}: {e}")

    return dict(zip(fai_df["chrom"], fai_df["length"]))

def compute_per_region_overlap(
    setA_df: pd.DataFrame,
    setB_df: pd.DataFrame,
    setB_index: Dict[str, Tuple[np.ndarray, np.ndarray]],
    chroms_include: Optional[set] = None
) -> np.ndarray:
    """
    Compute base-pair overlap between SetA and each individual region in SetB.

    Purpose:
        For each region in SetB, calculate how many base pairs overlap with any region in SetA.

    Reuses:
        - The binary search indexing strategy from compute_total_overlap().
        - The same setB_index = {chrom: (B_starts, B_ends)} structure.

    Returns:
        NumPy array of length len(SetB_df)
        Each element i corresponds to total bp overlap for SetB_df.iloc[i].
    """

    # Initialize per-region overlap accumulator
    overlap_per_B = np.zeros(len(setB_df), dtype=np.int64)

    # We’ll need quick lookup: map chrom → row indices for SetB extracts the row indices of each sub-group
    B_row_index = {
        chrom: group.index.to_numpy()
        for chrom, group in setB_df.groupby("chrom", sort=False)
    }

    for chrom, groupA in setA_df.groupby("chrom", sort=False): # iterate chrom: sub df dictionary
        if chroms_include and chrom not in chroms_include:
            continue
        if chrom not in setB_index:
            continue

        B_starts, B_ends = setB_index[chrom]
        B_idx = B_row_index[chrom]  # array of actual row indices in setB_df

        for a_start, a_end in zip(groupA["start"].values, groupA["end"].values):
            # Binary search range in sorted SetB starts
            i = np.searchsorted(B_starts, a_start, side="left")
            j = np.searchsorted(B_starts, a_end, side="left")

            cand_lo = max(0, i - 1)
            cand_hi = j

            b_s = B_starts[cand_lo:cand_hi]
            b_e = B_ends[cand_lo:cand_hi]
            b_rows = B_idx[cand_lo:cand_hi]  # which rows these belong to use B_idx

            # Compute overlap lengths vectorized
            left = np.maximum(a_start, b_s)
            right = np.minimum(a_end, b_e)
            overlap_bp = np.clip(right - left, a_min=0, a_max=None)

            # Accumulate per-B overlap
            overlap_per_B[b_rows] += overlap_bp # find row or b_idx at this candidate region, add this overlap to this space
            # array of len_b with overlap values of a at each index in Bp

    return overlap_per_B

def combined_permutation_tests(
    A_df: pd.DataFrame,
    B_df: pd.DataFrame,
    setB_index: Dict[str, Tuple[np.ndarray, np.ndarray]],
    chrom_lengths: dict,
    num_permutations: int,
    seed: int = 42,
) -> tuple[int, np.ndarray, float, pd.DataFrame]:
    """
    Run both global and per-region permutation tests.

    Returns
    -------
    observed_global : int
        Total observed overlap (SetA vs SetB).
    null_global : np.ndarray
        Array of total overlaps from all permutations.
    global_p : float
        One-tailed p-value for global overlap.
    per_region_df : pd.DataFrame
        Per-region results (with Bonferroni correction).
    """
    rng = np.random.default_rng(seed)

    # Initialize null distributions
    nB = len(B_df)
    null_global = np.empty(num_permutations, dtype=np.int64) # Allocate memory for storing permuted overlap ints
    null_per_region = np.zeros((num_permutations, nB), dtype=np.int64) 

    # Observed
    observed_global = compute_total_overlap(A_df, setB_index)
    observed_per_region = compute_per_region_overlap(A_df, B_df, setB_index)

    A_grouped = list(A_df.groupby("chrom", sort=False))

    for p in range(num_permutations):
        perm_A_rows = []

        for chrom, A_chr in A_grouped:
            chrom_len = chrom_lengths.get(chrom)
            if chrom_len is None or A_chr.empty:
                continue

            lengths = (A_chr["end"] - A_chr["start"]).to_numpy(np.int64)
            max_starts = chrom_len - lengths
            valid_mask = max_starts >= 0
            if not np.any(valid_mask):
                continue

            valid_lengths = lengths[valid_mask]
            valid_max_starts = max_starts[valid_mask]

            new_starts = np.fromiter(
                (rng.integers(0, hi + 1) for hi in valid_max_starts),
                dtype=np.int64,
                count=len(valid_max_starts),
            )
            new_ends = new_starts + valid_lengths

            perm_A_rows.append(pd.DataFrame({
                "chrom": chrom,
                "start": new_starts,
                "end": new_ends,
            }))

        perm_A_df = pd.concat(perm_A_rows, ignore_index=True)

        # Compute both overlaps using same permutation
        null_global[p] = compute_total_overlap(perm_A_df, setB_index)
        null_per_region[p, :] = compute_per_region_overlap(perm_A_df, B_df, setB_index) # Build out the per region null matrix

        '''
        if (p + 1) % 500 == 0:
            sys.stderr.write(f"Permutation {p + 1}/{num_permutations} done.\n")
        '''

    # Global test
    num_ge = np.sum(null_global >= observed_global)
    global_p = (num_ge + 1) / (num_permutations + 1) # compute global p val

    # Per-region test
    p_values = (np.sum(null_per_region >= observed_per_region, axis=0) + 1) / (num_permutations + 1)
    bonferroni_p = np.minimum(p_values * nB, 1.0)
    significant = bonferroni_p < 0.05

    per_region_df = B_df.copy()
    per_region_df["observed_overlap"] = observed_per_region
    per_region_df["p_value"] = p_values
    per_region_df["bonferroni_p"] = bonferroni_p # removed for turnin
    per_region_df["significant"] = significant

    return observed_global, null_global, global_p, per_region_df


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the permutation test script.

    Currently accepts:
      <setA_bed>   Path to SetA BED file (e.g., transcription factor binding sites)
      <setB_bed>   Path to SetB BED file (e.g., active chromatin regions)

    Placeholder arguments for:
      <genome_fai>  Genome index file (FAI)
      <output_dir>  Output directory
      [num_permutations]  (optional) number of random permutations

    Returns
    -------
    argparse.Namespace
        Object containing parsed arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Compute base-pair overlap between two BED files (SetA and SetB). "
            "Designed for permutation testing of genomic co-localization."
        )
    )

    # Required positional arguments (order matters)
    parser.add_argument("setA_bed", help="Path to SetA.bed (e.g., TF binding sites).")
    parser.add_argument("setB_bed", help="Path to SetB.bed (e.g., active chromatin regions).")
    parser.add_argument("genome_fai", help="Path to genome FASTA index (.fai).")
    parser.add_argument("output_dir", help="Directory to save results.")
    parser.add_argument("num_permutations", type=int, default=1000,
                        help="Number of random permutations (optional, default=1000).")

    return parser.parse_args()

def main():
    """Main entry point for the permutation test script."""
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    '''
    print("\n=== Genomic Overlap and Permutation Test ===")
    print(f"SetA file: {args.setA_bed}")
    print(f"SetB file: {args.setB_bed}")
    print(f"Genome index: {args.genome_fai}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of permutations: {args.num_permutations}")
    '''
    # -------------------------------------------------
    # 1: Load data
    # -------------------------------------------------
    setA_df = read_bed_as_dataframe(args.setA_bed)
    setB_df = read_bed_as_dataframe(args.setB_bed)
    setB_index = build_setB_index(setB_df)
    chr_len = load_fai(args.genome_fai)

    observed_global, null_global, global_p, per_region_df = combined_permutation_tests(
    setA_df, setB_df, setB_index, chr_len, args.num_permutations
    )

    # Count significant results
    sig_count = per_region_df["significant"].sum()
    bonf_threshold = 0.05 / len(setB_df)

    out_summary = Path(args.output_dir) / "results.tsv"
    out_per_region = Path(args.output_dir) / "results_per_region.tsv"

    # Prepare summary table
    summary_data = {
        "metric": [
            "observed_overlap",
            "global_p_value",
            "num_permutations",
            "setA_regions",
            "setB_regions",
            "setA_total_bases",
            "setB_total_bases",
            "bonferroni_threshold",
            "significant_regions_bonferroni"
        ],
        "value": [
            observed_global,
            round(global_p, 5),
            args.num_permutations,
            len(setA_df),
            len(setB_df),
            (setA_df["end"] - setA_df["start"]).sum(),
            (setB_df["end"] - setB_df["start"]).sum(),
            bonf_threshold,
            sig_count
        ]
    }
    pd.DataFrame(summary_data).to_csv(out_summary, sep="\t", index=False)

    # Sort per-region results by chromosome and start position
    per_region_df = per_region_df.sort_values(by=["chrom", "start"], kind="mergesort").reset_index(drop=True)

    # Rename columns and drop internal Bonferroni p-value column
    per_region_df.rename(columns={'significant': 'significant_bonferroni'}, inplace=True)
    per_region_df = per_region_df.drop('bonferroni_p', axis=1)

    # Write results to file
    per_region_df.to_csv(out_per_region, sep="\t", index=False)
    
    '''
    print(f"\n[INFO] Wrote summary to: {out_summary}")
    print(f"[INFO] Wrote per-region results to: {out_per_region}")
    print("\n=== Done ===\n")
    '''

if __name__ == "__main__":
    main()
