# BiocManager::install("IlluminaHumanMethylation450kanno.ilmn12.hg19")
# library(IlluminaHumanMethylation450kanno.ilmn12.hg19)
library(IlluminaHumanMethylation450kanno.ilmn12.hg19)
# 读你的数据
df <- read.table("/home/hjzhang/cancer/1bc818c4-d49b-4c73-b624-4ce818eee3e9.methylation_array.sesame.level3betas.txt", header=FALSE, sep="\t")
colnames(df) <- c("cg", "value")

# 加载注释
data(IlluminaHumanMethylation450kanno.ilmn12.hg19)
anno <- getAnnotation(IlluminaHumanMethylation450kanno.ilmn12.hg19)

# 合并
res <- merge(df, anno, by.x="cg", by.y="Name")

# 取关键列
res_out <- res[, c("cg", "chr", "pos", "value")]

write.table(res_out, "/home/hjzhang/cancer/1bc818c4_cg_hg19.txt", sep="\t", quote=FALSE, row.names=FALSE)