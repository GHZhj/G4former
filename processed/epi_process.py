import numpy as np
from concurrent.futures import ProcessPoolExecutor
import os
from itertools import islice

input_file = "/home/hjzhang/dataset/home-1/ylxiong/_update/Intervene_results/K562_1k/G4_1k.su2k_Z_D"
output_file = "/home/hjzhang/dataset/home-1/ylxiong/_update/Intervene_results/K562_1k/G4_1k.su2k_Z_D.tsv"


# 常量定义
SIGNAL_LENGTH = 2048
BATCH_SIZE = 5000
MAIN_CHR = 0
MAIN_START = 1
MAIN_END = 2
MAIN_STRAND = 3
SIG_CHR = 6         # 信号染色体
SIG_START = 7       # 信号起始
SIG_END = 8         # 信号结束
SIG_VALUE = 9       # 信号值
MIN_SIGNAL_LENGTH = 1  # 最小信号区间长度（可调整）


def process_batch(batch):
    batch_results = dict()
    for line_num, line in enumerate(batch, 1):
        fields = line.strip().split('\t')
        if len(fields) != 10:
            continue  # 列数错误，跳过
        
        # 解析主区域
        try:
            chr_main = fields[MAIN_CHR]
            start_main = int(fields[MAIN_START])
            end_main = int(fields[MAIN_END])
            strand_main = fields[MAIN_STRAND]
            key = (chr_main, start_main, end_main, strand_main)
            if end_main - start_main != SIGNAL_LENGTH:
                continue  # 主区域长度错误，跳过
        except (ValueError, IndexError):
            continue  # 主区域解析失败，跳过
        
        # 解析信号：先校验字段是否为空
        # 信号字段为空的情况：空字符串、'.'、'NA'等
        try:
            # 提取信号字段
            sig_chr_val = fields[SIG_CHR].strip()
            sig_start_val = fields[SIG_START].strip()
            sig_end_val = fields[SIG_END].strip()
            sig_value_val = fields[SIG_VALUE].strip()
            
            # 校验信号字段是否为空（任何一个为空则视为无效）
            if not all([sig_chr_val, sig_start_val, sig_end_val, sig_value_val]):
                # print(f"信号字段为空（{line_num}行）：{fields[SIG_CHR]},{fields[SIG_START]},{fields[SIG_END]},{fields[SIG_VALUE]}")
                # 信号字段为空，保留0值（若key不存在则初始化全零数组）
                if key not in batch_results:
                    batch_results[key] = np.zeros(SIGNAL_LENGTH, dtype=np.float32)
                continue  # 跳过后续处理
            
            # 字段非空，继续解析
            chr_sig = sig_chr_val
            start_sig = int(sig_start_val)
            end_sig = int(sig_end_val)
            value_sig = float(sig_value_val)
            
            # 校验信号区间长度
            signal_length = end_sig - start_sig
            if signal_length < MIN_SIGNAL_LENGTH:
                # print(f"信号区间过短（{line_num}行）：长度{signal_length} < {MIN_SIGNAL_LENGTH}")
                if key not in batch_results:
                    batch_results[key] = np.zeros(SIGNAL_LENGTH, dtype=np.float32)
                continue
            
            # 校验信号位置有效性
            if chr_sig != chr_main or start_sig < start_main or end_sig > end_main:
                # print(f"信号位置无效（{line_num}行）：主区域[{start_main},{end_main})，信号[{start_sig},{end_sig})")
                if key not in batch_results:
                    batch_results[key] = np.zeros(SIGNAL_LENGTH, dtype=np.float32)
                continue
        
        except (ValueError, IndexError):
            # 信号解析失败（如非数值类型），保留0值
            if key not in batch_results:
                batch_results[key] = np.zeros(SIGNAL_LENGTH, dtype=np.float32)
            continue
        
        # 计算有效索引并赋值（仅处理有效信号）
        indices = np.arange(start_sig, end_sig, dtype=np.int32) - start_main
        valid_indices = indices[(indices >= 0) & (indices < SIGNAL_LENGTH)]
        
        sig_arr = np.zeros(SIGNAL_LENGTH, dtype=np.float32)
        if valid_indices.size > 0:
            sig_arr[valid_indices] = value_sig
        
        # 合并信号（取最大值，空信号或无效信号已被跳过，保留0值）
        if key in batch_results:
            np.maximum(batch_results[key], sig_arr, out=batch_results[key])
        else:
            batch_results[key] = sig_arr
    
    return batch_results


def batch_generator(filepath, batch_size=BATCH_SIZE):
    with open(filepath, 'r') as f:
        while True:
            batch = list(islice(f, batch_size))
            if not batch:
                break
            yield batch


def main():
    max_workers = min(os.cpu_count() * 2, 200)
    print(f"使用 {max_workers} 个进程处理...")

    region_signal_map = dict()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for batch_dict in executor.map(process_batch, batch_generator(input_file), chunksize=2):
            for key, sig_arr in batch_dict.items():
                if key in region_signal_map:
                    np.maximum(region_signal_map[key], sig_arr, out=region_signal_map[key])
                else:
                    region_signal_map[key] = sig_arr

    # 写入结果（保留0值）
    with open(output_file, 'w') as f:
        lines = []
        total_regions = len(region_signal_map)
        non_zero_regions = 0
        for key, sig_arr in region_signal_map.items():
            chr_main, start_main, end_main, strand = key
            signal_str = '\t'.join(f"{v:.6f}" for v in sig_arr)
            lines.append(f"{chr_main}\t{start_main}\t{end_main}\t{strand}\t{signal_str}\n")
            if np.any(sig_arr != 0):
                non_zero_regions += 1
        
        f.writelines(lines)
        print(f"处理完成：共 {total_regions} 个区域，{non_zero_regions} 个区域有非零信号，其余保留0值")
        print(f"结果保存至 {output_file}")


if __name__ == '__main__':
    main()