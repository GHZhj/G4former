# 加载必要的库
library(tidyverse)
library(patchwork)
library(showtext)

# ==========================================
# 1. 数据读取与解析
# ==========================================
csv_file <- "../test6.csv"
df <- read.csv(csv_file, check.names = FALSE)
colnames(df)[1] <- "Model" 

df_long <- df %>%
  pivot_longer(cols = -Model, names_to = "TestSet", values_to = "Raw") %>%
  mutate(kv_pairs = str_extract_all(Raw, "([a-zA-Z0-9 ]+)\\s*=\\s*([0-9.]+)")) %>%
  unnest(kv_pairs) %>%
  separate(kv_pairs, into = c("Metric", "Value"), sep = "=") %>%
  mutate(
    Metric = trimws(Metric),
    Value = as.numeric(Value),
    Metric = case_when(
      Metric == "Test Acc" ~ "Acc",
      Metric == "AUPRC" ~ "AUPRC",
      TRUE ~ Metric
    )
  ) %>%
  # ⭐ 过滤阶段就去掉 MCC 和 r
  filter(Metric %in% c("AUC", "Acc", "F1", "AUPRC"))

# ==========================================
# 2. 颜色与绘图配置
# ==========================================
plot_colors <- c("#003D7C", "#80679C", "#C01020", "#FF9900", "#8E9124", "#33A02C")

create_bar_plot <- function(data, target_metric, y_lims) {
  plot_df <- data %>% 
    filter(Metric == target_metric, TestSet == "K562") %>%
    mutate(
      line_type = ifelse(str_detect(Model, "-DNase"), "dashed", "blank"),
      border_col = ifelse(str_detect(Model, "-DNase"), "Yes", "No")
    )
  
  ggplot(plot_df, aes(x = Model, y = Value, fill = Model)) +
    geom_col(aes(color = border_col, linetype = line_type), 
             width = 0.7, 
             size = 0.8) +
    geom_text(aes(label = sprintf("%.4f", Value)), 
              vjust = -0.5, size = 5.5, fontface = "bold") +
    # ⭐ 添加图例标题
    scale_fill_manual(values = plot_colors, name = "Models") +
    scale_color_manual(values = c("Yes" = "#FF0000", "No" = "transparent"), guide = "none") +
    scale_linetype_identity(guide = "none") + 
    coord_cartesian(ylim = y_lims) +
    labs(title = target_metric, x = NULL, y = NULL) +
    theme_bw() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 20),
      axis.text.x = element_text(angle = 45, hjust = 1, size = 12, face = "bold", color = "black"),
      axis.text.y = element_text(size = 18, face = "bold", color = "black"),
      panel.grid.major.y = element_line(color = "grey80", linetype = "dashed", size = 0.2),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      legend.position = "none", # ⭐ 如需显示图例请保持 "right"，不需要则设为 "none"
      panel.border = element_rect(color = "black", fill = NA, size = 0.2)
    )
}

# ==========================================
# 3. 生成并组合 (仅保留 4 个核心子图)
# ==========================================
p_auc  <- create_bar_plot(df_long, "AUC",  c(0.70, 0.98))
p_acc  <- create_bar_plot(df_long, "Acc",  c(0.75, 0.92))
p_f1   <- create_bar_plot(df_long, "F1",   c(0.55, 0.85))
p_aupr <- create_bar_plot(df_long, "AUPRC", c(0.65, 0.95))

# ⭐ 布局改为 2x2，并收集图例
final_plot <- (p_auc | p_acc) / (p_f1 | p_aupr) + 
  plot_layout(guides = 'collect') + 
  plot_annotation(
    title = "Test on K562",
    theme = theme(
      plot.title = element_text(size = 22, face = "bold", hjust = 0.5, margin = margin(b = 20))
    )
  )

# ==========================================
# 4. 保存 (由于子图变少，调整了宽高比)
# ==========================================
ggsave("DNase_4Metrics_K562.pdf", final_plot, width = 10, height = 10, dpi = 300)
print(final_plot)