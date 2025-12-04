# create_strong_signal.py
import os

def write_file(path, lines):
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

outdir = "strong_signal_test"
os.makedirs(outdir, exist_ok=True)

# genome
write_file(os.path.join(outdir, "genome.fai"), ["chr1\t10000"])

# SetA: many small intervals packed inside 1000-1200
setA_lines = []
pos = 1000
for i in range(50):
    setA_lines.append(f"chr1\t{pos}\t{pos+4}")
    pos += 4  # pack tightly
write_file(os.path.join(outdir, "SetA.bed"), setA_lines)

# SetB: one target region that spans those A peaks, plus 4 other distractors
setB_lines = [
    "chr1\t1000\t1200",
    "chr1\t2000\t2100",
    "chr1\t3000\t3100",
    "chr1\t4000\t4100",
    "chr1\t5000\t5100",
]
write_file(os.path.join(outdir, "SetB.bed"), setB_lines)

print("Wrote files to:", outdir)
