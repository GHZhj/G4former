# 加载必要的库
library(tidyverse)
library(patchwork)
library(showtext)

# ==========================================
# 1. 数据读取与解析
# ==========================================

csv_file <- "../test7.csv"
df <- read.csv(csv_file, check.names = FALSE)
colnames(df)[1] <- "Size" 

df_long <- df %>%
  mutate(Size = as.factor(Size)) %>% # 将 Size 转为因子用于分组
  pivot_longer(cols = -Size, names_to = "TestSet", values_to = "Raw") %>%
  mutate(kv_pairs = str_extract_all(Raw, "([a-zA-Z0-9 ]+)\\s*=\\s*([0-9.]+)")) %>%
  unnest(kv_pairs) %>%
  separate(kv_pairs, into = c("Metric", "Value"), sep = "=") %>%
  mutate(
    Metric = trimws(Metric),
    Value = as.numeric(Value),
    Metric = case_when(
      Metric == "Test Acc" ~ "Acc",
      Metric == "AUPRCC" ~ "AUPRC",
      TRUE ~ Metric
    )
  ) %>%
  filter(Metric %in% c("AUC", "Acc", "F1", "AUPRC"))

# ==========================================
# 2. 颜色与绘图配置
# ==========================================
# 为不同 Size 设置颜色
size_colors <- c("1" = "#C01020", "100" = "#42d1d7") 

create_grouped_plot <- function(data, target_metric, y_lims) {
  plot_df <- data %>% filter(Metric == target_metric)
  
  ggplot(plot_df, aes(x = TestSet, y = Value, fill = Size)) +
    # 使用 position_dodge 让柱子并列
    geom_col(position = position_dodge(width = 0.8), 
             width = 0.7, 
             color = "black", 
             size = 0.3) +
    # ⭐ 修改这里：添加 angle = 45
    # hjust = 0 可以让倾斜文字的起点对齐柱子顶部
    geom_text(aes(label = sprintf("%.4f", Value)), 
              position = position_dodge(width = 0.8),
              vjust = -0.8,    # 稍微调高一点，避免倾斜后重叠
              hjust = 0,    # 微调水平位置
              angle = 45,     # ⭐ 倾斜 45 度
              size = 5.5,     # 调大数字字体
              fontface = "bold") +
    scale_fill_manual(values = size_colors, name = "Resolution Size") +
    # ⭐ 建议：因为数值倾斜会占用上方空间，可以稍微调高 y 轴上限
    coord_cartesian(ylim = c(y_lims[1], y_lims[2] )) +
    labs(title = target_metric, x = "Test on cell types", y = NULL) +
    theme_bw() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 18),
      axis.text.x = element_text(size = 20, face = "bold", color = "black"),
      axis.text.y = element_text(size = 20, face = "bold", color = "black"),
      panel.grid.major.y = element_line(color = "grey80", linetype = "dashed", size = 0.2),
      panel.grid.major.x = element_blank(),
      legend.position = "right",
      legend.key.height = unit(1.0, "cm"),
      legend.spacing.y = unit(0.5, "cm"),
      legend.title = element_text(size = 20, face = "bold"),
      legend.text = element_text(size = 20, face = "bold"),
      panel.border = element_rect(color = "black", fill = NA, size = 0.5)
    )
}

# ==========================================
# 3. 生成 2x2 组合图
# ==========================================
p_auc  <- create_grouped_plot(df_long, "AUC",  c(0.70, 1.00))
p_Acc  <- create_grouped_plot(df_long, "Acc",  c(0.70, 1.00))
p_f1   <- create_grouped_plot(df_long, "F1",   c(0.50, 0.90))
p_AUPRC <- create_grouped_plot(df_long, "AUPRC", c(0.60, 0.95))

final_plot <- (p_auc | p_Acc) / (p_f1 | p_AUPRC) + 
  plot_layout(guides = 'collect') + 
  plot_annotation(
    title = "Comparison of ATAC resolution",
    theme = theme(
      plot.title = element_text(size = 22, face = "bold", hjust = 0.5, margin = margin(b = 20))
    )
  )

# ==========================================
# 4. 保存
# ==========================================
ggsave("Size_Comparison_Results.pdf", final_plot, width = 10, height = 10, dpi = 300)
print(final_plot)