#!/usr/bin/env Rscript

library(DESeq2)
library(tidyverse)

# =========================
# 1. 读取数据
# =========================
tumor_file  <- "/home/hjzhang/THCA/4413cb28-32a4-4ae8-b4c5-6bfe3f19f869.rna_seq.augmented_star_gene_counts.tsv"
normal_file <- "/home/hjzhang/THCA/_cd83494b-8644-4db6-8b05-ec642a6a8dca.rna_seq.augmented_star_gene_counts.tsv"

tumor  <- read.table(tumor_file, header=TRUE, sep="\t", comment.char="#", check.names=FALSE)
normal <- read.table(normal_file, header=TRUE, sep="\t", comment.char="#", check.names=FALSE)

# =========================
# 2. 去掉非基因行（N_开头）
# =========================
tumor  <- tumor[!grepl("^N_", tumor$gene_id), ]
normal <- normal[!grepl("^N_", normal$gene_id), ]

# =========================
# 3. 提取count（用 unstranded）
# =========================
# =========================
tumor_counts  <- tumor[, c("gene_id", "gene_name", "unstranded")]
normal_counts <- normal[, c("gene_id", "unstranded")]

colnames(tumor_counts)[3]  <- "tumor"
colnames(normal_counts)[2] <- "normal"

# =========================
# 4. 合并
# =========================
df <- inner_join(tumor_counts, normal_counts, by="gene_id")

# 转矩阵
count_matrix <- as.matrix(df[, c("tumor", "normal")])
rownames(count_matrix) <- df$gene_id

# 转整数（DESeq2要求）
count_matrix <- round(count_matrix)

# =========================
# 5. 样本信息
# =========================
sample_info <- data.frame(
  row.names = c("tumor", "normal"),
  condition = c("Tumor", "Normal")
)

sample_info$condition <- factor(sample_info$condition,
                                levels=c("Normal", "Tumor"))

# =========================
# 6. 构建DESeq对象
# =========================
dds <- DESeqDataSetFromMatrix(
  countData = count_matrix,
  colData = sample_info,
  design = ~ condition
)

# 过滤低表达
dds <- dds[rowSums(counts(dds)) >= 10, ]

# =========================
# 🚨 关键：单样本情况处理
# =========================
dds <- estimateSizeFactors(dds)

# 手动设置 dispersion（因为没有replicate）
dispersions(dds) <- rep(0.1, nrow(dds))

# =========================
# 7. 差异分析
# =========================
dds <- nbinomWaldTest(dds)

res <- results(dds)

# =========================
# 8. 整理结果
# =========================
deg <- res %>%
  as.data.frame() %>%
  rownames_to_column("gene_id") %>%
  left_join(df[, c("gene_id", "gene_name")], by="gene_id") %>%
  filter(!is.na(log2FoldChange))
# 排序
deg <- deg[order(deg$log2FoldChange, decreasing=TRUE), ]

# =========================
# 9. 分类（更可靠）
# =========================
deg$group <- "Stable"
deg$group[deg$log2FoldChange > 1]  <- "Up"
deg$group[deg$log2FoldChange < -1] <- "Down"

# =========================
# 10. 保存
# =========================
write.csv(deg, "/home/hjzhang/THCA/DESeq2_results_all.csv", row.names=FALSE)
write.csv(deg[deg$group=="Up", ], "/home/hjzhang/THCA/DESeq2_up.csv", row.names=FALSE)
write.csv(deg[deg$group=="Down", ], "/home/hjzhang/THCA/DESeq2_down.csv", row.names=FALSE)

cat("完成：\n")
cat("总基因数:", nrow(deg), "\n")
cat("上调:", sum(deg$group=="Up"), "\n")
cat("下调:", sum(deg$group=="Down"), "\n")