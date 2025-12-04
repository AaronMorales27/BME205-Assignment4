"""
Compute the total number of base pairs of overlap between two genomic region sets (SetA and SetB),
counting each region in SetA independently.

Implements  assignment pseudocode:
    For each region in SetA:
        Identify candidate overlapping regions in SetB (same chromosome) using binary search.
        Compute overlap in base pairs for each candidate.
        Accumulate total overlap.

No merging of overlapping intervals within either set is performed.

Dependencies:
    numpy  - efficient numeric computation, vectorized overlap calculation
    pandas - fast parsing of BED files, simple grouping by chromosome
    standard library modules (argparse, sys, os) only

ASSUMPTIONS VERY IMPORTANT   
    BED files need to be congruent in naming style, same exact column names

    Make sure both BEDs are 0-based, half-open (the BED standard).
    If one uses 1-based inclusive coordinates (like GTFs), you'll get near-zero overlaps.
"""

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

'''
df = read_bed_as_dataframe('data/SetA.bed')
print(df.head(10)) # SANITY CHECK, unsorted .bed dataframe
'''

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

### PART 2 ###

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

# Sanity Check
'''
dict = load_fai('data/genome.fa.fai')
print(dict)
'''

def permute_and_test(
    A_df: pd.DataFrame,
    # B_df: pd.DataFrame,
    setB_index: Dict[str, Tuple[np.ndarray, np.ndarray]],
    chrom_lengths: dict,
    num_permutations: int, # ADD ARGUEMENT LATER
    seed: int = 42,
) -> tuple[int, np.ndarray, float]:
    """
    Perform permutation test for genomic region overlap.

    Parameters
    ----------
    A_df : pd.DataFrame
        Set A (regions to randomize).
    B_df : pd.DataFrame
        Set B (reference regions).
    chrom_lengths : dict
        Mapping {chrom: chromosome_length}.
    compute_overlap_fn : callable
        Function that returns total bp overlap between two BED DataFrames.
        It should accept either (A_chr_df, B_chr_df) for single-chromosome DataFrames
        or full DataFrames (the function will be called per-chromosome in this script).
    num_permutations : int
        Number of random permutations.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    observed_overlap : int
        Base-pair overlap of original A vs B.
    null_overlaps : np.ndarray
        Overlap values from random permutations.
    p_value : float
        One-tailed p-value.
    """
  
    rng = np.random.default_rng(seed)
    null_overlaps = np.empty(num_permutations, dtype=np.int64) # Allocate memory for storing permuted overlap ints

    # Determine chromosomes to test: those present in A (we place A on its chromosomes),
    # but use B's intervals when available (missing B on chrom => overlap 0)
    # chroms = sorted(set(A_df["chrom"].unique()))

    '''
    Don't need to group our dataframes
    # Group A and B by chromosome for efficiency (dictionary of DataFrames)
    A_by_chr = dict(tuple(A_df.groupby("chrom")))
    B_by_chr = dict(tuple(B_df.groupby("chrom")))
    chroms = sorted(A_by_chr.keys())
    # Compute observed overlap using provided overlap function (per-chromosome summation) RETROFIT with b_index and total overal functions
    observed_overlap = 0
    for c in chroms:
        a_chr = A_by_chr.get(c)
        b_chr = B_by_chr.get(c, pd.DataFrame(columns=["chrom", "start", "end"])) # If chrom = c not in B_by_chr dict, return empty df w/ cols
        if a_chr is None or a_chr.empty:
            continue
        # compute_overlap_fn is expected to return integer bp overlap between two DataFrames
    '''
    observed_overlap = compute_total_overlap(A_df, setB_index)

    # Perform permutations
    A_grouped = list(A_df.groupby("chrom", sort=False))

    for p in range(num_permutations):
        total_perm_overlap = 0
        '''
        for chrom in chroms:
            A_chr = A_by_chr[chrom]
            B_chr = B_by_chr.get(chrom, pd.DataFrame(columns=["chrom", "start", "end"]))
            chrom_len = chrom_lengths.get(chrom)
            if chrom_len is None or A_chr.empty:
                # If chromosome length is unknown, skip placements on this chromosome
                continue
        '''
    
        for chrom, A_chr in A_grouped:
        # for chrom, A_chr in A_df.groupby("chrom", sort=False): # iterate dictionary of chrom: sub df, by chrom and A_df
            chrom_len = chrom_lengths.get(chrom) # chrome length for this chrom
            if chrom_len is None or A_chr.empty:
                continue

            # region lengths
            lengths = (A_chr["end"] - A_chr["start"]).to_numpy(np.int64) # pandas vectorized subtract 2 series = numpy array of lengths
            max_starts = chrom_len - lengths # chrom_length is an integer length for this chrom, subtract this by each element in lengths
            valid_mask = max_starts >= 0 # Boolean masked array of same size as lengths, T value if region fits in chrom (not out of bounds)

            if not np.any(valid_mask): # if only false/ out of bounds regions skip this chrom
                # All regions on this chrom are longer than the chromosome; skip
                continue

            # Keep only regions that can be placed
            valid_lengths = lengths[valid_mask] # apply boolean mask to filter valid lengths
            valid_max_starts = max_starts[valid_mask] # filter valid max start positions(chrom_len - lengths)

            # Generate random start positions per region (preserving size)
            # Note: RNG cannot vectorize different 'high' per-element, so use generator
            new_starts = np.fromiter( # doesn’t need to materialize the entire list in memory first fromiter efficient for large datasets
                (rng.integers(0, hi + 1) for hi in valid_max_starts), # iterates random values from 0 - max start(inclusive)
                dtype=np.int64,
                count=len(valid_max_starts), # iterates for each valid max_start value
            ) # CAN NOT VECTORIZE rng here as our bounds are highly variable
            new_ends = new_starts + valid_lengths # respective numpy array of random valid ends aswell

            perm_A_chr = pd.DataFrame({
                "chrom": chrom,
                "start": new_starts,
                "end": new_ends,
            }) # Doesn't need to be sorted for computing overlap, not when we have B_df_index

            total_perm_overlap += compute_total_overlap(perm_A_chr, setB_index)

        null_overlaps[p] = total_perm_overlap # place overlap value into null_overlaps array
        '''
        # Optional lightweight progress reporting
        if (p + 1) % 50 == 0 or (p + 1) == num_permutations:
            print(f"Permutation {p + 1}/{num_permutations} done. Current perm overlap: {total_perm_overlap:,}")
        '''
        if (p + 1) % 10 == 0:
            sys.stderr.write(f"{p + 1}/{num_permutations} permutations done. Current perm overlap: {total_perm_overlap:,}\n")

    # One-tailed p-value (permutation test with +1 correction)
    num_ge = int(np.sum(null_overlaps >= observed_overlap)) # num of random chance overlaps exceeding observed
    p_value = (num_ge + 1) / (num_permutations + 1) # + 1 to avoid zero p vals

    return observed_overlap, null_overlaps, p_value

### PART 3 ###
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
            overlap_per_B[b_rows] += overlap_bp

    return overlap_per_B

def per_region_permutation_test(
    A_df: pd.DataFrame,
    B_df: pd.DataFrame,
    setB_index: Dict[str, Tuple[np.ndarray, np.ndarray]],
    chrom_lengths: dict,
    num_permutations: int,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compute per-region permutation test for SetB regions.

    Returns
    -------
    results_df : pd.DataFrame with columns:
        ['chrom', 'start', 'end', 'observed_overlap', 'p_value', 'bonferroni_p', 'significant']
    """
    rng = np.random.default_rng(seed)
    nB = len(B_df)
    null_matrix = np.zeros((num_permutations, nB), dtype=np.int64)

    # Observed overlap per SetB region
    observed = compute_per_region_overlap(A_df, B_df, setB_index)

    # Pre-group SetA by chromosome once
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

            perm_A_rows.append(pd.DataFrame({ # list of small DataFrames, one per chromosome, each with permuted A regions.
                "chrom": chrom,
                "start": new_starts,
                "end": new_ends,
            }))

        perm_A_df = pd.concat(perm_A_rows, ignore_index=True) # merges all the chromosome-specific DataFrames (perm_A_rows) into a single large DataFrame (perm_A_df).
        perm_overlap = compute_per_region_overlap(perm_A_df, B_df, setB_index)
        null_matrix[p, :] = perm_overlap # Fill in the n permute x n_b intervals, pth permute with perm_overlap for the column entries

        if (p + 1) % 10 == 0:
            sys.stderr.write(f"Per-region permutation {p + 1}/{num_permutations} done.\n")

    # Compute p-values (one-tailed)
    p_values = (np.sum(null_matrix >= observed, axis=0) + 1) / (num_permutations + 1) # 1D numpy array of len nB with respective p vals
    # sum the number of permuted overlaps that are larger than the true num permutations

    # Bonferroni correction
    # p val < 0.05 / trials(or snps)
    bonferroni_p = np.minimum(p_values * nB, 1.0) # Multiply each p-value by nB and compare to 0.05, some may exceed 1.0 so clip them
    significant = bonferroni_p < 0.05 # boolean mask indices with significance  < 0.05/trials ( the correction )
    # multiply bonferroni_p by nb instead of 0.5/nb
    results_df = B_df.copy()
    results_df["observed_overlap"] = observed
    results_df["p_value"] = p_values
    results_df["bonferroni_p"] = bonferroni_p
    results_df["significant"] = significant


    print("Observed per-B overlaps (first 10):", observed[:10])
    print("Mean perm overlaps (first 10):", null_matrix.mean(axis=0)[:10])
    print("Raw p-values (first 10):", p_values[:10])
    print("Bonferroni (first 10):", bonferroni_p[:10])

    return results_df

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

        if (p + 1) % 500 == 0:
            sys.stderr.write(f"Permutation {p + 1}/{num_permutations} done.\n")

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
    per_region_df["bonferroni_p"] = bonferroni_p
    per_region_df["significant"] = significant

    print("\n=== Global permutation test ===")
    print(f"Observed overlap: {observed_global:,} bp")
    print(f"Global p-value: {global_p:.6f}")

    print("\n=== Per-region permutation summary ===")
    print(f"Number of B regions: {nB}")
    print(f"Bonferroni threshold: {0.05 / nB:.2e}")
    print(f"Significant regions: {significant.sum()} / {nB}")

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

    print("\n=== Genomic Overlap and Permutation Test ===")
    print(f"SetA file: {args.setA_bed}")
    print(f"SetB file: {args.setB_bed}")
    print(f"Genome index: {args.genome_fai}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of permutations: {args.num_permutations}")

    # -------------------------------------------------
    # 1: Load data
    # -------------------------------------------------
    setA_df = read_bed_as_dataframe(args.setA_bed)
    setB_df = read_bed_as_dataframe(args.setB_bed)
    setB_index = build_setB_index(setB_df)
    chr_len = load_fai(args.genome_fai)
    '''
    # -------------------------------------------------
    # 2: Global overlap + permutation test
    # -------------------------------------------------
    observed_overlap, null_overlaps, global_p_value = permute_and_test(
        setA_df, setB_index, chr_len, args.num_permutations
    )

    print(f"\n[RESULT] Global overlap = {observed_overlap:,} bp")
    print(f"[RESULT] Global p-value = {global_p_value:.5f}")

    # -------------------------------------------------
    # 3: Per-region significance analysis 
    # -------------------------------------------------
    results_df = per_region_permutation_test(
        setA_df, setB_df, setB_index, chr_len, args.num_permutations
    )

    # Count significant results
    sig_count = results_df["significant"].sum()
    bonf_threshold = 0.05 / len(setB_df)

    print(f"\n[RESULT] Bonferroni threshold = {bonf_threshold:.8f}")
    print(f"[RESULT] Significant SetB regions = {sig_count}/{len(setB_df)}")

    # -------------------------------------------------
    # 4️⃣ Output results
    # -------------------------------------------------
    out_summary = Path(args.output_dir) / "results.tsv"
    out_per_region = Path(args.output_dir) / "results_per_region.tsv"
    '''


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

    # Save per-region details
    per_region_df.to_csv(out_per_region, sep="\t", index=False)

    print(f"\n[INFO] Wrote summary to: {out_summary}")
    print(f"[INFO] Wrote per-region results to: {out_per_region}")
    print("\n=== Done ===\n")


if __name__ == "__main__":
    main()
