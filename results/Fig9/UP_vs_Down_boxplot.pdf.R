# ==============================
# 0. 加载包
# ==============================
library(tidyverse)
library(ggplot2)

# ==============================
# 1. 读取数据
# ==============================
df <- read.csv("../test8.csv")

# 转宽表
df_wide <- df %>%
  pivot_wider(names_from = expression, values_from = both)

# 去掉总体行（如果有）
df_wide <- df_wide %>%
  filter(name != "cancer")

# ==============================
# 2. 转成长格式（用于画图）
# ==============================
df_long <- df_wide %>%
  pivot_longer(cols = c(UP, Down),
               names_to = "group",
               values_to = "value")

# ==============================
# 3. 去异常值（IQR 方法）
# ==============================
remove_outliers <- function(x) {
  q1 <- quantile(x, 0.25)
  q3 <- quantile(x, 0.75)
  iqr <- q3 - q1
  x >= (q1 - 1.5 * iqr) & x <= (q3 + 1.5 * iqr)
}

df_filtered <- df_long %>%
  group_by(group) %>%
  filter(remove_outliers(value)) %>%
  ungroup()

# ==============================
# 4. 统计检验（Wilcoxon）
# ==============================
test_res <- wilcox.test(value ~ group, data = df_filtered)

p_value <- test_res$p.value

# ==============================
# 5. 相关性分析（可选，论文推荐一起报）
# ==============================
cor_res <- cor.test(df_wide$UP, df_wide$Down, method = "spearman")

R_value <- cor_res$estimate
cor_p <- cor_res$p.value

cat("Spearman R =", R_value, "\n")
cat("Correlation p =", cor_p, "\n")
cat("Wilcoxon p =", p_value, "\n")

# ==============================
# 6. 作图（柱状 + 散点 + P值）
# ==============================
p <- ggplot(df_filtered, aes(x = group, y = value, fill = group)) +
  
  # 柱状图（均值）
  stat_summary(fun = mean, geom = "bar", width = 0.6, color = "black") +
  
  # 误差线（SE）
  stat_summary(fun.data = mean_se, geom = "errorbar", width = 0.2) +
  
  # 散点
  geom_jitter(width = 0.1, size = 2, color = "black") +
  
  # 颜色（和你图一致）
  scale_fill_manual(values = c("UP" = "#4C78A8", "Down" = "#F58518")) +
  
  theme_classic(base_size = 14) +
  labs(
    x = "",
    y = "Number of DEGs"
  ) +
  
  # P值横线
  geom_segment(
    aes(x = 1, xend = 2,
        y = max(df_filtered$value) * 1.05,
        yend = max(df_filtered$value) * 1.05),
    inherit.aes = FALSE
  ) +
  
  # P值文字
  annotate(
    "text",
    x = 1.5,
    y = max(df_filtered$value) * 1.12,
    label = paste0("P = ", signif(p_value, 3)),
    size = 5
  )

print(p)

# ==============================
# 7. 保存图片
# ==============================
ggsave("UP_vs_Down_boxplot.pdf", p, width = 5, height = 5)