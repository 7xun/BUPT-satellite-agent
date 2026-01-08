# -*- coding: utf-8 -*-
"""
阿里云 OSS 单文件上传脚本
用于上传 oss/processed_data 下的单个文件到 OSS 对应目录 (E/{DATA_ID}/...)
"""
import os
import sys
import argparse

try:
    import oss2
except ImportError:
    print("错误: 未找到 oss2 模块。")
    print("请在终端运行以下命令安装阿里云 OSS SDK:")
    print("pip install oss2")
    sys.exit(1)

# =============================================================================
# 1. 配置信息
# =============================================================================

ACCESS_KEY_ID = os.environ.get('OSS_ACCESS_KEY_ID', 'YOUR_ACCESS_KEY_ID')
ACCESS_KEY_SECRET = os.environ.get('OSS_ACCESS_KEY_SECRET', 'YOUR_ACCESS_KEY_SECRET')
ENDPOINT = 'oss-cn-beijing.aliyuncs.com'
BUCKET_NAME = 'hede-satellite'

# =============================================================================
# 2. 上传逻辑
# =============================================================================

def upload_single_file(local_path):
    # 1. 路径检查与解析
    local_path = os.path.abspath(local_path)
    if not os.path.exists(local_path):
        print(f"❌ 错误: 文件不存在: {local_path}")
        return

    if not os.path.isfile(local_path):
        print(f"❌ 错误: 路径不是一个文件: {local_path}")
        return

    # 尝试解析 DATA_ID (即 processed_data 下的一级目录名)
    # 假设路径结构包含 .../processed_data/{DATA_ID}/{filename}
    parts = local_path.split(os.sep)
    
    try:
        # 找到 processed_data 的索引
        # 注意：Windows下路径分隔符可能是 \，split(os.sep) 应该能处理
        # 为了更稳健，可以先统一转为 / 再 split，或者倒序查找
        
        if 'processed_data' in parts:
            idx = parts.index('processed_data')
            if idx + 2 < len(parts): # 确保后面有 DATA_ID 和 filename
                data_id = parts[idx + 1]
                filename = parts[-1]
            elif idx + 1 < len(parts): # 可能是直接在 processed_data 下的文件? (不符合当前结构，但做个防御)
                 # 如果结构是 processed_data/filename，没有中间层
                 # 根据用户描述 "processed_data/0x0821这个文件下的csv"，应该是有中间层的
                 print(f"❌ 错误: 无法从路径中解析出 DATA_ID。预期路径结构: .../processed_data/{{DATA_ID}}/{{filename}}")
                 return
            else:
                print(f"❌ 错误: 路径结构异常。")
                return
        else:
            # 如果不在 processed_data 目录下，尝试手动指定或报错
            # 这里简单处理：报错
            print(f"❌ 错误: 文件不在 'processed_data' 目录下，无法自动推断 OSS 目标路径。")
            print(f"当前路径: {local_path}")
            return

    except ValueError:
        print(f"❌ 错误: 路径解析失败。")
        return

    # 构造 OSS 路径: E/{DATA_ID}/{filename}
    oss_path = f"E/{data_id}/{filename}"

    print(f"正在连接 OSS (Bucket: {BUCKET_NAME})...")
    
    try:
        auth = oss2.Auth(ACCESS_KEY_ID, ACCESS_KEY_SECRET)
        bucket = oss2.Bucket(auth, ENDPOINT, BUCKET_NAME)

        print(f"🚀 开始上传:")
        print(f"   本地: {local_path}")
        print(f"   目标: oss://{BUCKET_NAME}/{oss_path}")
        
        oss2.resumable_upload(
            bucket, 
            oss_path, 
            local_path,
            store=oss2.ResumableStore(root='/tmp'), 
            multipart_threshold=10 * 1024 * 1024,
            part_size=1 * 1024 * 1024,
            num_threads=4
        )
        
        print("✅ 上传成功！")
        
    except Exception as e:
        print(f"❌ 上传失败: {e}")

if __name__ == '__main__':
    # 默认示例文件 (方便直接运行测试)
    # 假设当前脚本在 oss/ 目录下，向上找 processed_data
    default_file = os.path.join(os.path.dirname(__file__), 'processed_data', '0x0821', '2023_07.csv')
    
    parser = argparse.ArgumentParser(description='上传单个文件到 OSS')
    parser.add_argument('file_path', nargs='?', default=default_file, help='要上传的本地文件路径')
    
    args = parser.parse_args()
    
    print(f"运行参数: {args.file_path}")
    upload_single_file(args.file_path)
