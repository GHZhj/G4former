library(seqinr)
library(pqsfinder)
library(rtracklayer)
library(Biostrings)
library(BiocParallel)

# Process command line arguments
args = commandArgs(trailingOnly=TRUE)
argsLen <- length(args);
if (argsLen != 4) {
  stop("Specify the fasta file, output file, minimum score, and overlapping flag (1 or 0, for True or False)", call.=FALSE)
}

FastaFile = paste0(args[1])
OutputFile = paste0(args[2])
MinScore = as.numeric(paste0(args[3]))
Overlapping = as.numeric(paste0(args[4]))

cat(sprintf("FastaFile: %s\n", FastaFile))
cat(sprintf("OutputFile: %s\n", OutputFile))
cat(sprintf("MinScore: %s\n", MinScore))
cat(sprintf("Overlapping: %s\n", Overlapping))

cat("Reading the fasta file...\n")
seq_list <- read.fasta(FastaFile, as.string=TRUE)

# ✅ 定义每条序列的处理函数
process_sequence <- function(seq_name) {
  seq <- paste(seq_list[[seq_name]], collapse = "")
  seq <- toupper(seq)

  dnaString <- DNAString(seq)
  print(seq)
  # 确保 min_score 都是正数
  score_thresholds <- c(MinScore, MinScore - 10, MinScore - 20 ,1)
  score_thresholds <- score_thresholds[score_thresholds > 0]
  if (length(score_thresholds) == 0) score_thresholds <- 0

  best_pqs <- NULL
  best_score <- -Inf
  # best_strands <- -Inf

  for (th in score_thresholds) {
    pqs <- tryCatch(
      pqsfinder(dnaString, min_score = th),
      error = function(e) NULL
    )

    if (!is.null(pqs) && length(pqs) > 0) {
      scores <- score(pqs)
      max_idx <- which.max(scores)

      if (scores[max_idx] > best_score) {
        best_pqs <- pqs[max_idx]
        best_score <- scores[max_idx]
        # best_strands <- strand(best_pqs)[max_idx]

      }
      if (!is.null(best_pqs)) break
    }
  }

  if (!is.null(best_pqs)) {
      # 转成 DNAStringSet 对象
      best_seq <- as(best_pqs, "DNAStringSet")

      # 获取原始 header
      headers <- names(best_seq)

      # 用 seq_name 替换 pqsfinder
      headers <- sub("^pqsfinder", seq_name, headers)

      # 设置新 header
      names(best_seq) <- headers

      return(best_seq)
    } else {
      return(NULL)
    }
}
# ✅ 并行执行
n_cores <- max(1, parallel::detectCores() )
cat(sprintf("Running pqsfinder in parallel using %d cores...\n", n_cores))

# macOS/Linux 用 MulticoreParam；Windows 改为 SnowParam
param <- MulticoreParam(workers = n_cores)
results <- bplapply(names(seq_list), process_sequence, BPPARAM = param)

# ✅ 合并结果
valid_results <- results[!vapply(results, is.null, logical(1))]
all_pqs <- do.call(c, valid_results)
print(all_pqs)

cat("Exporting pqsfinder output...\n")
writeXStringSet(as(all_pqs,"DNAStringSet"), filepath = OutputFile, format = "fasta")

cat("Done!\n")

# Rscript pqsfinder.R originPeak.fa originPeak.fasta 40 "1"
# FastaFile: originPeak.fa.fa
# OutputFile: originPeak.fa.fasta
# MinScore: 40
# Overlapping: 1
# Reading the fasta file...
# Running pqsfinder in parallel using 47 cores...
