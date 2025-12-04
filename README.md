# Assignment: Genomic Region Overlap Analysis and Permutation Testing

## Description

You are a computational biologist investigating the relationship between transcription factor binding sites and regulatory chromatin regions. You will implement a permutation test to determine whether the overlap between two sets of genomic regions is statistically significant, a fundamental technique in genomics and regulatory biology.

This analysis examines potential co-localization between:
- **Set A**: Transcription factor binding sites identified by ChIP-seq
- **Set B**: Active chromatin regions identified by histone modification mapping

## Data Files Provided

You will be given the following files:

1. **BED files**: Two sets of genomic coordinates in BED format
   - `SetA.bed` - Transcription factor binding sites (3-column BED format)
   - `SetB.bed` - Active chromatin regions (3-column BED format)
   - Each BED file contains chromosome, start position, and end position

2. **Genome index**: `genome.fa.fai` - Chromosome lengths in FASTA index format

## Assignment Tasks

### Part 1: Overlap Calculation (40 points)

Implement efficient genomic interval overlap detection with interval trees or similar data structures:

**Overlap Calculation Method:**

For each region in Set A, calculate the number of base pairs that overlap with any region in Set B. Sum across all Set A regions.

**CRITICAL: Each interval must be considered independently. Do not merge overlapping intervals within Set A or Set B.**

**Algorithm:**
```
total_overlap = 0
for each region_A in SetA:
    for each region_B in SetB:
        if same_chromosome:
            overlap_bp = calculate_overlap(region_A, region_B)
            total_overlap += overlap_bp
return total_overlap
```

**Example:**
```
SetA: chr1:100-200, chr1:150-250 (two overlapping regions)
SetB: chr1:120-180

Correct calculation:
  Region A1 (100-200) ∩ B1 (120-180) = 60 bp
  Region A2 (150-250) ∩ B1 (120-180) = 30 bp
  Total weighted overlap = 90 bp

INCORRECT (if you merge SetA first):
  Merged SetA: chr1:100-250
  (100-250) ∩ (120-180) = 60 bp
  Total = 60 bp  ❌ WRONG!
```

**Biological Rationale:** Overlapping ChIP-seq peaks within Set A represent independent measurements indicating stronger or more reproducible transcription factor binding. Regions with multiple overlapping peaks are high-confidence binding sites. We want to test whether active chromatin regions (Set B) preferentially overlap with these high-confidence binding sites.

**Requirements:**
- Count overlapping base pairs, not overlapping regions
- Implement efficient interval overlap algorithms (interval trees recommended for large datasets)
- Process all chromosomes present in the data
- Preserve overlapping structure within each set

### Part 2: Permutation Testing (40 points)

Perform statistical significance testing using permutation analysis to generate a global p-value:

1. **Implement permutation strategy**: Randomly redistribute Set A regions while preserving:
   - Number of regions
   - Size of each individual region
   - Chromosome assignment (regions stay on their original chromosome)

2. **Generate null distribution**: Create 10,000 random permutations
3. **Calculate global p-value**: Compare observed overlap to null distribution
4. **Handle edge cases**: Avoid region placement outside chromosome boundaries

**Permutation Algorithm:**
- For each region in Set A, randomly select a new start position on the same chromosome
- Preserve the region's length (if original region is 500bp, permuted region is 500bp)
- Ensure the region doesn't extend beyond chromosome boundaries
- Calculate weighted overlap for each permuted Set A against the original Set B

**P-value Calculation:**
```python
# Count permutations with overlap >= observed
num_greater_or_equal = sum(1 for perm_overlap in null_distribution 
                           if perm_overlap >= observed_overlap)

# Calculate one-tailed p-value
p_value = (num_greater_or_equal + 1) / (num_permutations + 1)
```

### Part 3: Per-Region Analysis and Multiple Testing Correction (20 points)

Calculate significance for individual regions and apply Bonferroni correction:

1. **Calculate per-region p-values**: For each region in Set B, calculate how many base pairs overlap with Set A regions
2. **Generate per-region null distributions**
3. **Apply Bonferroni correction**: 
4. **Report significant regions**:

**Bonferroni Correction:**

When testing multiple hypotheses (one per Set B region), we risk false positives. The Bonferroni correction controls for this:

```python
bonferroni_threshold = 0.05 / num_setB_regions

# A region is significant if:
if p_value < bonferroni_threshold:
    region_is_significant = True
```

**Why correction is needed:** You are performing thousands of statistical tests (one per Set B region). Without correction, you would expect approximately 5% of regions to show p < 0.05 by random chance alone, leading to many false positives.


## Technical Requirements

### Command Line Interface

Your script must accept command line arguments:

```bash
python firstname_lastname_permutation_test.py <setA_bed> <setB_bed> <genome_fai> <output_dir> [num_permutations]
```

Example:
```bash
python firstname_lastname_permutation_test.py data/SetA.bed data/SetB.bed data/genome.fa.fai . 10000
```

This will generate two output files:
- `results.tsv` - Summary statistics
- `results_per_region.tsv` - Per-region results for all Set B regions

### Required Environment

Your script must run in the provided conda environment: `permutation_test.yaml`

You may use the following Python libraries (all included in the environment):
- **numpy** - for numerical operations and random number generation
- **pandas** - for data manipulation and file output
- Standard library modules (random, sys, argparse, etc.)
- **DO NOT** use specialized bioinformatics libraries (pybedtools, pysam, etc.)

### Output Files

#### Summary Output (results.tsv)

Your script must generate a TSV file with the following format:

```
metric	value
observed_overlap	345279
global_p_value	0.0234
num_permutations	10000
setA_regions	1245
setB_regions	2156
setA_total_bases	512847
setB_total_bases	891234
bonferroni_threshold	0.000023
significant_regions_bonferroni	42
```

**Required metrics:**
- `observed_overlap` - Total weighted overlapping bases between Set A and Set B
- `global_p_value` - Statistical significance from permutation test (global overlap)
- `num_permutations` - Number of permutations performed
- `setA_regions` - Number of regions in Set A
- `setB_regions` - Number of regions in Set B
- `setA_total_bases` - Total bases covered by Set A (sum of all region lengths)
- `setB_total_bases` - Total bases covered by Set B (sum of all region lengths)
- `bonferroni_threshold` - Adjusted significance threshold (0.05 / number of Set B regions)
- `significant_regions_bonferroni` - Number of Set B regions with p-value < bonferroni_threshold

#### Per-Region Output (results_per_region.tsv)

Your script must also generate a per-region results file:

```
chrom	start	end	observed_overlap	p_value	significant_bonferroni
chr1	1000	2000	150	0.0012	True
chr1	5000	5500	200	0.0001	True
chr2	3000	3400	50	0.3421	False
chr2	8000	8600	0	1.0000	False
```

**Required columns:**
- `chrom` - Chromosome name from Set B region
- `start` - Start position of Set B region
- `end` - End position of Set B region
- `observed_overlap` - Number of base pairs this Set B region overlaps with Set A
- `p_value` - Per-region p-value from permutation test
- `significant_bonferroni` - Boolean (True/False) indicating if p-value < bonferroni_threshold

**Notes:**
- Sort regions by chromosome then start position
- Include ALL Set B regions, even those with 0 overlap
- Use tab-separated format with header row

## Submission Requirements

Submit the following files:

1. **Python script**: `firstname_lastname_permutation_test.py`
2. **Validation document**: `firstname_lastname_validation.pdf` (1 paragraph)

Your script must:
- Run in the provided `permutation_test.yaml` conda environment
- Accept the required command line arguments
- Generate both output files (summary and per-region)


## Data Format Notes

### BED File Format
- **3-column format**: chromosome, start, end (tab-separated)
- **0-based coordinates**: Start position is inclusive, end position is exclusive


### FASTA Index Format
- **5-column format**: chromosome, length, offset, bases_per_line, bytes_per_line
- **Only need**: chromosome name (column 1) and length (column 2)
- **Example**: `chr1	249250621	52	50	51`

## Performance Requirements

- **Runtime limit**: Your program must complete within a reasonable time for 10,000 permutations on the provided test datasets
- **Memory efficiency**: Handle large datasets without excessive memory usage
- **Algorithm efficiency**: Use interval trees or similar data structures for efficient overlap queries

**Note:** Naive nested loop approaches will exceed the time limit on large datasets. You must implement efficient data structures.

## Tips and Hints

1. **Start small**: Test your overlap calculation on small examples first
2. **Use bedtools to validate**: Compare your overlap calculation against `bedtools intersect -wo`
3. **Implement interval trees**: For large datasets, nested loops will be too slow
4. **Test incrementally**: Verify each part works before moving to the next
