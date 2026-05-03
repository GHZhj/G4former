import re

input_file = "../../processed/originPeak.fasta"
output_file = "_Ourdataset_score.bed"

with open(input_file) as fin, open(output_file, "w") as fout:
    for line in fin:
        if line.startswith(">"):
            # 例: >chr1:10287-10309(-);G_quartet;...;score=109;
            chrom_match = re.search(r'(chr[\w\d]+):(\d+)-(\d+)', line)
            score_match = re.search(r'score=(\d+)', line)
            if chrom_match and score_match:
                chrom, start, end = chrom_match.groups()
                score = score_match.group(1)
                fout.write(f"{chrom}\t{start}\t{end}\t{score}\n")

print(f"✅ 已保存 BED 文件到: {output_file}")