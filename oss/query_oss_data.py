# -*- coding: utf-8 -*-
"""
从 OSS 查询 CSV 数据并导出为 JSON
"""
import duckdb
import json
import os
import pandas as pd

# =============================================================================
# 配置信息 (请与上传脚本保持一致)
# =============================================================================
ACCESS_KEY_ID = os.environ.get('OSS_ACCESS_KEY_ID', 'YOUR_ACCESS_KEY_ID')
ACCESS_KEY_SECRET = os.environ.get('OSS_ACCESS_KEY_SECRET', 'YOUR_ACCESS_KEY_SECRET')
ENDPOINT = 'oss-cn-beijing.aliyuncs.com'
BUCKET_NAME = 'hede-satellite'
OSS_OBJECT_KEY = 'data/0x0821_.csv' # OSS 上的文件路径

# 查询参数
# 纳秒时间戳 (对应 2024-10-13 00:00:00.664)
START_TIME = 1728748800664000000
# 纳秒时间戳 (对应 2024-10-13 00:00:10.664)
END_TIME = 1728748838664000000
# 输出文件路径：保存到当前脚本所在目录下的 data 文件夹中
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "data", "solar_current_data.json")

def query_oss_csv():
    # 1. 构建 OSS 完整路径 (s3://bucket/key)
    # 阿里云 OSS 兼容 S3 协议
    s3_path = f"s3://{BUCKET_NAME}/{OSS_OBJECT_KEY}"
    
    print(f"正在连接 OSS 并查询: {s3_path}")
    print(f"时间范围(纳秒): {START_TIME} -> {END_TIME}")

    con = duckdb.connect()
    
    try:
        # 2. 加载 httpfs 扩展并配置 OSS 凭证
        con.execute("INSTALL httpfs;")
        con.execute("LOAD httpfs;")
        con.execute(f"SET s3_endpoint='{ENDPOINT}';")
        con.execute(f"SET s3_access_key_id='{ACCESS_KEY_ID}';")
        con.execute(f"SET s3_secret_access_key='{ACCESS_KEY_SECRET}';")
        # con.execute("SET s3_url_style='path';") # 注释掉此行，使用默认的 Virtual-Host 模式
        con.execute("SET s3_region='oss-cn-beijing';") # 显式设置区域

        # 3. 执行 SQL 查询
        # 使用 time 列 (纳秒时间戳) 进行过滤
        query = f"""
            SELECT 
                "time",
                "ZTMS001-太阳电池阵1电流" as current1,
                "ZTMS002-太阳电池阵2电流" as current2,
                "ZTMS003-太阳电池阵3电流" as current3
            FROM read_csv_auto('{s3_path}')
            WHERE "time" >= {START_TIME} AND "time" <= {END_TIME}
        """
        
        df = con.sql(query).df()
        
        if df.empty:
            print("⚠️ 未查询到数据，请检查时间范围或文件路径。")
            return

        print(f"✅ 查询成功，共找到 {len(df)} 条记录。")

        # 4. 导出为 JSON
        # orient='records' 生成 [{"time":..., "current1":...}, ...] 格式
        json_str = df.to_json(orient='records', force_ascii=False, indent=4)
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(json_str)
            
        print(f"✅ 数据已保存至: {os.path.abspath(OUTPUT_FILE)}")
        
        # 打印前几条预览
        print("\n数据预览:")
        print(df.head())

    except Exception as e:
        print(f"❌ 查询失败: {e}")
    finally:
        con.close()

if __name__ == '__main__':
    query_oss_csv()
