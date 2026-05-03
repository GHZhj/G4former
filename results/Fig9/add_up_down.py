import pandas as pd

# 1. 读取差异表达结果文件
# 注意：如果你的 CSV 是用逗号分隔的，read_csv 默认即可处理
deg_df = pd.read_csv('/home/hjzhang/THCA/DESeq2_results_all.csv')

# 2. 读取基因坐标文件
# 假设 gene_simple.txt 是空格或制表符分隔，且没有表头
# 我们手动指定列名为 ['chr', 'start', 'end', 'gene_id']
coords_df = pd.read_csv('/home/hjzhang/ACS/DESeq2/gene_simple.txt', sep=r'\s+', header=None, 
                        names=['chr', 'start', 'end', 'gene_id'])

# 3. 执行合并 (Merge)
# 使用 'left' join 可以确保保留 pseudo_DEG_results.csv 中的所有基因
# 即使某些基因在 gene_simple.txt 中找不到坐标，也会填充为 NaN
result_df = pd.merge(deg_df, coords_df, on='gene_id', how='left')

# 4. 优化列顺序（可选）
# 把坐标信息挪到 gene_id 后面，方便查看
cols = ['gene_id', 'chr', 'start', 'end'] + [c for c in deg_df.columns if c != 'gene_id']
result_df = result_df[cols]

# 5. 保存结果
result_df.to_csv('/home/hjzhang/THCA/DESeq2_results_all_with_coords.csv', index=False)

print("合并完成！结果已保存至 DEG_with_coords.csv")
print(result_df.head())


import pandas as pd

# 1. 加载你合并后的数据 (假设文件名为 merged_results.csv)
df = pd.read_csv('/home/hjzhang/THCA/DESeq2_results_all_with_coords.csv')

# 2. 定义你想要保留的列
# 注意：这里包含了 group，因为我们需要用它来过滤
selected_columns = ['chr', 'start', 'end', 'gene_name', 'log2FoldChange','group']
subset_df = df[selected_columns]

# 3. 按照 group 拆分并保存
# 常见的 group 有: 'Down', 'Stable', 'Up'
groups = subset_df['group'].unique()

for g in groups:
    # 过滤出当前组的数据
    group_data = subset_df[subset_df['group'] == g].copy()
    
    # 移除 group 列，只保留你要求的四个坐标相关列
    final_output = group_data[['chr', 'start', 'end', 'gene_name','log2FoldChange']]
    
    # 保存为 CSV 或文本文件 (这里建议保存为 .bed 格式，Tab分隔，无表头，生信通用)
    file_name = f"/home/hjzhang/THCA/genes_{g}.bed"
    final_output.to_csv(file_name, sep='\t', index=False, header=False)
    
    print(f"已生成组别 [{g}] 的文件: {file_name} (包含 {len(final_output)} 个基因)")
    