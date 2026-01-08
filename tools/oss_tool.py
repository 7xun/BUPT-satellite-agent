import os
import pandas as pd
from langchain_core.tools import tool
from config import (
    OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET, OSS_ENDPOINT, OSS_BUCKET_NAME,
    OUTPUT_ROOT, PARQUET_ROOT as DATA_ROOT
)

try:
    import oss2
except ImportError:
    oss2 = None

def _download_from_oss(oss_path, local_path):
    if not oss2:
        return False, "oss2 module not installed"
    
    try:
        auth = oss2.Auth(OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET)
        bucket = oss2.Bucket(auth, OSS_ENDPOINT, OSS_BUCKET_NAME)
        
        if not bucket.object_exists(oss_path):
            return False, f"OSS file not found: {oss_path}"
            
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        bucket.get_object_to_file(oss_path, local_path)
        return True, "Download success"
    except Exception as e:
        return False, str(e)

@tool
def query_oss_csv_data(satellite: str, bag_id: str, year: str, week: str, column: str, value: str) -> str:
    """
    从OSS下载指定卫星、包、时间的CSV数据，并筛选特定列的值。
    参数:
    - satellite: 卫星代号 (如 "E")
    - bag_id: 包ID (如 "0x0821")
    - year: 年份 (如 "2023")
    - week: 周数 (如 "7" 或 "07")
    - column: 要筛选的列名 (如 "ZTMS015-帆板1状态")
    - value: 目标值 (如 "未展开")
    """
    try:
        # 1. 格式化参数
        week_str = f"{int(week):02d}"
        file_name = f"{year}_{week_str}.csv"
        
        # OSS 路径: E/0x0821/2023_07.csv
        oss_key = f"{satellite}/{bag_id}/{file_name}"
        
        # 本地路径: data/E/0x0821/2023_07.csv
        local_dir = os.path.join(DATA_ROOT, satellite, bag_id)
        local_path = os.path.join(local_dir, file_name)
        
        # 2. 检查本地文件或下载
        if not os.path.exists(local_path):
            print(f"Downloading {oss_key} to {local_path}...")
            success, msg = _download_from_oss(oss_key, local_path)
            if not success:
                return f"无法获取数据文件: {msg}"
        
        # 3. 读取并筛选
        try:
            df = pd.read_csv(local_path, encoding='utf-8')
        except:
            try:
                df = pd.read_csv(local_path, encoding='gb18030')
            except:
                df = pd.read_csv(local_path, encoding='ISO-8859-1')
        
        if column not in df.columns:
            return f"列名 '{column}' 不存在。可用列: {list(df.columns)[:5]}..."
            
        # 筛选
        mask = df[column].astype(str) == str(value)
        result_df = df[mask].copy()
        
        if result_df.empty:
            return f"未找到符合条件的数据 ({column} == {value})。"
            
        # 格式化时间列
        if 'time' in result_df.columns:
            try:
                # 尝试转换时间列
                if pd.api.types.is_numeric_dtype(result_df['time']):
                    # 假设是纳秒时间戳
                    result_df['time'] = pd.to_datetime(result_df['time'], unit='ns')
                else:
                    # 尝试自动解析字符串
                    result_df['time'] = pd.to_datetime(result_df['time'])
                
                # 格式化为字符串: YYYY-MM-DD HH:MM:SS.mmm
                result_df['time'] = result_df['time'].dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]
            except Exception as e:
                print(f"Warning: Time column conversion failed: {e}")

        # 4. 保存结果
        output_filename = f"{satellite}_{bag_id}_{year}_{week_str}_filtered.json"
        output_path = os.path.join(OUTPUT_ROOT, output_filename)
        os.makedirs(OUTPUT_ROOT, exist_ok=True)
        
        result_df.to_json(output_path, orient='records', force_ascii=False, indent=4)
        
        return f"查询成功，找到 {len(result_df)} 条数据。结果已保存至: {output_path}"
        
    except Exception as e:
        return f"处理数据时出错: {e}"
