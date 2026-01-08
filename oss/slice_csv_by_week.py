# -*- coding: utf-8 -*-
"""
按周切分 CSV 文件脚本
"""
import pandas as pd
import os

# =============================================================================
# 配置
# =============================================================================
# 输入文件路径 (假设在项目根目录)
INPUT_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), '0x0821.csv')

# 输出目录 (在 oss 文件夹下)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'processed_data', '0x0821')

def slice_csv_by_week():
    # 1. 检查输入文件
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误: 找不到输入文件: {INPUT_FILE}")
        # 尝试查找带下划线的文件名，以防万一
        alt_file = INPUT_FILE.replace('.csv', '_.csv')
        if os.path.exists(alt_file):
            print(f"⚠️ 找到替代文件: {alt_file}，将使用此文件。")
            input_path = alt_file
        else:
            return
    else:
        input_path = INPUT_FILE

    print(f"正在读取文件: {input_path} ...")
    
    # 4. 创建输出目录 (提前创建)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"已创建输出目录: {OUTPUT_DIR}")
    
    # 用于记录本次运行已创建的文件，避免追加到旧文件
    initialized_files = set()
    total_rows = 0
    chunk_size = 100000  # 每次处理 10 万行

    try:
        # 2. 分块读取 CSV
        reader = pd.read_csv(input_path, chunksize=chunk_size)
        
        for i, df in enumerate(reader):
            if 'time' not in df.columns:
                print("❌ 错误: CSV 文件中缺少 'time' 列。")
                return

            print(f"正在处理第 {i+1} 个分块 (行数: {len(df)})...")
            total_rows += len(df)

            # 3. 处理时间列
            # 假设 time 列是纳秒时间戳 (int64)
            # 如果是字符串格式，pd.to_datetime 也能智能识别
            df['temp_datetime'] = pd.to_datetime(df['time'], unit='ns')
            
            # 提取 ISO 年份和周数
            # 注意：使用 isocalendar 可以避免跨年周的问题
            iso_cal = df['temp_datetime'].dt.isocalendar()
            df['year'] = iso_cal.year
            df['week'] = iso_cal.week
            
            # 5. 分组并保存 (追加模式)
            grouped = df.groupby(['year', 'week'])
            
            for (year, week), group in grouped:
                # 构造文件名: YYYY_WW.csv (例如 2023_08.csv)
                filename = f"{year}_{week:02d}.csv"
                filepath = os.path.join(OUTPUT_DIR, filename)
                
                # 删除辅助列，保持原样输出
                output_df = group.drop(columns=['temp_datetime', 'year', 'week'])
                
                # 确定写入模式和是否写入表头
                if filepath not in initialized_files:
                    mode = 'w' # 第一次写入该文件，覆盖旧文件
                    header = True
                    initialized_files.add(filepath)
                    print(f"   -> 创建/覆盖文件: {filename}")
                else:
                    mode = 'a' # 后续写入，追加
                    header = False
                
                output_df.to_csv(filepath, index=False, encoding='utf-8-sig', mode=mode, header=header)

        print(f"\n🎉 处理完成！共处理 {total_rows} 行数据。")
        print(f"文件保存在: {OUTPUT_DIR}")

    except Exception as e:
        print(f"❌ 处理过程中发生错误: {e}")

if __name__ == '__main__':
    slice_csv_by_week()
