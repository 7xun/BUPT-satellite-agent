# -*- coding: utf-8 -*-
"""
OSS 数据异常分析脚本
根据指定的列名和异常值，从 OSS 上的 CSV 文件中筛选数据并导出为 JSON。
"""
import os
import sys
import argparse
import json
import pandas as pd
import io

try:
    import oss2
except ImportError:
    print("错误: 未找到 oss2 模块。")
    print("请在终端运行: pip install oss2")
    sys.exit(1)

# =============================================================================
# 配置信息
# =============================================================================
ACCESS_KEY_ID = os.environ.get('OSS_ACCESS_KEY_ID', 'YOUR_ACCESS_KEY_ID')
ACCESS_KEY_SECRET = os.environ.get('OSS_ACCESS_KEY_SECRET', 'YOUR_ACCESS_KEY_SECRET')
ENDPOINT = 'oss-cn-beijing.aliyuncs.com'
BUCKET_NAME = 'hede-satellite'

def analyze_abnormal_data(oss_file, columns, abnormal_value, compare_columns=None, output_file=None):
    print(f"正在连接 OSS (Bucket: {BUCKET_NAME})...")
    
    try:
        # 1. 连接 OSS
        auth = oss2.Auth(ACCESS_KEY_ID, ACCESS_KEY_SECRET)
        bucket = oss2.Bucket(auth, ENDPOINT, BUCKET_NAME)

        # 检查文件是否存在
        if not bucket.object_exists(oss_file):
            print(f"❌ 错误: OSS 上找不到文件: {oss_file}")
            return

        print(f"正在读取文件: {oss_file} ...")
        
        # 2. 读取 CSV 文件
        # 使用 get_object 获取流，直接传给 pandas
        # 注意：如果文件非常大，建议下载到本地再处理，或者使用 chunksize
        obj = bucket.get_object(oss_file)
        # 读取内容到内存 (BytesIO)，方便 pandas 处理 seek 等操作 (虽然 read_csv 支持流，但 BytesIO 更稳)
        content = obj.read()
        
        try:
            # 优先尝试 UTF-8 (包含 utf-8-sig)
            df = pd.read_csv(io.BytesIO(content), encoding='utf-8')
        except UnicodeDecodeError:
            print("⚠️ UTF-8 解码失败，尝试使用 GB18030 (兼容 GBK) 解码...")
            try:
                df = pd.read_csv(io.BytesIO(content), encoding='gb18030')
            except UnicodeDecodeError:
                print("❌ GB18030 解码也失败了，尝试使用 ISO-8859-1 (可能导致乱码)...")
                df = pd.read_csv(io.BytesIO(content), encoding='ISO-8859-1')
        
        print(f"文件读取成功，共 {len(df)} 行。正在筛选数据...")

        # 3. 筛选数据
        # 逻辑：只要指定的任意一列的值等于 abnormal_value，就选中该行
        mask = pd.Series(False, index=df.index)
        found_columns = []
        
        for col in columns:
            col = col.strip() # 去除可能的多余空格
            if col in df.columns:
                # 注意数据类型，将列转为字符串进行比较，防止类型不匹配
                mask = mask | (df[col].astype(str) == str(abnormal_value))
                found_columns.append(col)
            else:
                print(f"⚠️ 警告: 文件中未找到列 '{col}'，已跳过。")

        if not found_columns:
            print("❌ 错误: 未找到任何有效的列名，无法进行筛选。")
            print(f"文件中的列名: {list(df.columns)}")
            return

        result_df = df[mask].copy()
        # 添加原始索引列，方便后续对应
        result_df['original_index'] = result_df.index
        
        print(f"筛选完成: 找到 {len(result_df)} 行异常数据 (条件: {found_columns} == '{abnormal_value}')")

        # 准备输出目录
        project_root = os.path.dirname(os.path.dirname(__file__))
        output_dir = os.path.join(project_root, 'output')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        base_name = os.path.basename(oss_file) # 2023_07.csv
        name_without_ext = os.path.splitext(base_name)[0] # 2023_07

        # 4. 导出 JSON 结果
        if output_file is None:
            output_file = os.path.join(output_dir, f"{name_without_ext}_abnormal.json")

        # 保存为 JSON
        # orient='records' 生成列表形式: [{"col1": val1, ...}, ...]
        # force_ascii=False 保证中文正常显示
        result_df.to_json(output_file, orient='records', force_ascii=False, indent=4)
        
        print(f"✅ 异常数据已保存至: {output_file}")

        # 5. 上下文数据提取逻辑 (修改后)
        if compare_columns:
            print(f"正在提取上下文数据 (目标列: {compare_columns}, 范围: 前后10行)...")
            
            # 确保目标列存在
            valid_compare_cols = [c for c in compare_columns if c in df.columns]
            if not valid_compare_cols:
                print(f"⚠️ 警告: 未找到任何有效的对比列，跳过上下文提取。")
            else:
                abnormal_indices = df[mask].index
                context_results = []

                for idx in abnormal_indices:
                    # 假设默认 RangeIndex，idx 即为位置
                    # 使用 iloc 获取前后 10 行
                    start_pos = max(0, idx - 10)
                    end_pos = min(len(df), idx + 11) # 包含 idx 本身，及后面 10 行
                    
                    # 提取数据
                    subset = df.iloc[start_pos:end_pos].copy()
                    
                    # 筛选列
                    subset = subset[valid_compare_cols]
                    
                    # 将 subset 转换为字典列表，保留索引以便知道是哪一行
                    # reset_index 将索引变为列，方便查看行号
                    subset_with_index = subset.reset_index()
                    # 重命名 index 列为 original_index 以保持一致性
                    if 'index' in subset_with_index.columns:
                        subset_with_index.rename(columns={'index': 'original_index'}, inplace=True)
                        
                    subset_records = subset_with_index.to_dict(orient='records')
                    
                    # 构建结构
                    context_item = {
                        "ref_abnormal_idx": int(idx),
                        "context_data": subset_records
                    }
                    context_results.append(context_item)
                
                if context_results:
                    context_output_file = os.path.join(output_dir, f"{name_without_ext}_context.json")
                    # 保存为 JSON
                    with open(context_output_file, 'w', encoding='utf-8') as f:
                        json.dump(context_results, f, ensure_ascii=False, indent=4)
                    print(f"✅ 上下文数据已保存至: {context_output_file} (共 {len(context_results)} 组)")
                else:
                    print("ℹ️ 未提取到任何上下文数据。")

    except Exception as e:
        print(f"❌ 处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # 默认测试参数
    DEFAULT_FILE = 'E/0x0821/2023_07.csv'
    DEFAULT_COLUMNS = ['ZTMS015-帆板1状态']
    DEFAULT_VALUE = '未展开'
    DEFAULT_COMPARE_COLUMNS = ['ZTMS026-X轮通信状态']

    parser = argparse.ArgumentParser(description='OSS 数据异常分析脚本')
    
    # 参数 (设置默认值，不再强制必填)
    parser.add_argument('--file', default=DEFAULT_FILE, help=f'OSS 文件路径 (默认: {DEFAULT_FILE})')
    parser.add_argument('--columns', nargs='+', default=DEFAULT_COLUMNS, help=f'要检查的列名 (默认: {DEFAULT_COLUMNS})')
    parser.add_argument('--value', default=DEFAULT_VALUE, help=f'异常状态值 (默认: {DEFAULT_VALUE})')
    
    # 新增参数：对比列
    parser.add_argument('--compare_columns', nargs='+', default=DEFAULT_COMPARE_COLUMNS, help=f'需要对比的列名 (默认: {DEFAULT_COMPARE_COLUMNS})')

    # 可选参数
    parser.add_argument('--output', help='输出 JSON 文件路径 (默认保存在 output 目录下)')

    args = parser.parse_args()
    
    print(f"运行参数:")
    print(f"  - 文件: {args.file}")
    print(f"  - 列名: {args.columns}")
    print(f"  - 异常值: {args.value}")
    print(f"  - 对比列: {args.compare_columns}")

    analyze_abnormal_data(args.file, args.columns, args.value, args.compare_columns, args.output)
