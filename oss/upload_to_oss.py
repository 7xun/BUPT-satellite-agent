# -*- coding: utf-8 -*-
"""
阿里云 OSS 上传脚本
使用前请确保已安装 oss2 库: pip install oss2
"""
import os
import sys

try:
    import oss2
except ImportError:
    print("错误: 未找到 oss2 模块。")
    print("请在终端运行以下命令安装阿里云 OSS SDK:")
    print("pip install oss2")
    sys.exit(1)

# =============================================================================
# 1. 配置信息 (请在此处填入你的阿里云 OSS 信息)
# =============================================================================

# [必填] 阿里云账号 AccessKey
# 建议使用 RAM 用户 AccessKey，获取方式：https://ram.console.aliyun.com/users
ACCESS_KEY_ID = os.environ.get('OSS_ACCESS_KEY_ID', 'YOUR_ACCESS_KEY_ID')
ACCESS_KEY_SECRET = os.environ.get('OSS_ACCESS_KEY_SECRET', 'YOUR_ACCESS_KEY_SECRET')

# [必填] Bucket 所在地域的 Endpoint
# 例如北京是: oss-cn-beijing.aliyuncs.com
# 可以在 OSS 控制台的 Bucket 概览页面找到
ENDPOINT = 'oss-cn-beijing.aliyuncs.com'

# [必填] Bucket 名称
BUCKET_NAME = 'hede-satellite'

# [配置] 数据 ID (用于匹配本地文件夹和 OSS 文件夹)
DATA_ID = '0x0821'

# 本地文件夹路径: oss/processed_data/0x0821
LOCAL_DIR = os.path.join(os.path.dirname(__file__), 'processed_data', DATA_ID)

# OSS 目标文件夹路径: E/0x0821
OSS_PREFIX = f'E/{DATA_ID}'

# =============================================================================
# 2. 上传逻辑
# =============================================================================

def upload_directory():
    # 检查配置是否已填写
    if ACCESS_KEY_ID == '你的AccessKeyId' or BUCKET_NAME == '你的Bucket名称':
        print("❌ 请先打开此脚本，修改配置信息 (AccessKey, Endpoint, BucketName)！")
        return

    # 检查本地文件夹是否存在
    if not os.path.exists(LOCAL_DIR):
        print(f"❌ 找不到本地文件夹: {LOCAL_DIR}")
        return

    print(f"正在连接 OSS (Bucket: {BUCKET_NAME})...")
    
    try:
        # 创建 Bucket 对象
        auth = oss2.Auth(ACCESS_KEY_ID, ACCESS_KEY_SECRET)
        bucket = oss2.Bucket(auth, ENDPOINT, BUCKET_NAME)

        # 获取文件列表
        files = [f for f in os.listdir(LOCAL_DIR) if f.endswith('.csv')]
        if not files:
            print(f"⚠️  文件夹 {LOCAL_DIR} 中没有 CSV 文件。")
            return

        print(f"📂 准备上传 {len(files)} 个文件到 oss://{BUCKET_NAME}/{OSS_PREFIX}/ ...")

        success_count = 0
        for i, filename in enumerate(files):
            local_path = os.path.join(LOCAL_DIR, filename)
            oss_path = f"{OSS_PREFIX}/{filename}"
            
            print(f"[{i+1}/{len(files)}] 正在上传: {filename} ...", end='', flush=True)
            
            # 使用断点续传接口
            oss2.resumable_upload(
                bucket, 
                oss_path, 
                local_path,
                store=oss2.ResumableStore(root='/tmp'), 
                multipart_threshold=10 * 1024 * 1024,
                part_size=1 * 1024 * 1024,
                num_threads=4
            )
            print(" ✅")
            success_count += 1
        
        print(f"\n🎉 全部完成！成功上传 {success_count} 个文件。")
        
    except Exception as e:
        print(f"\n❌ 上传过程中发生错误: {e}")

if __name__ == '__main__':
    upload_directory()
