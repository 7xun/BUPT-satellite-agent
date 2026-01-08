# -*- coding: utf-8 -*-
"""
全局配置模块
"""
import os

DASHSCOPE_API_KEY = "sk-af0bd19b890d465ea71ca754cf2a6658"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

QWEN_MODEL = "qwen-plus"
EMBEDDING_MODEL = "text-embedding-v3"
EMBEDDING_DIMENSIONS = None

# 路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VECTOR_STORE_DIR = os.path.join(PROJECT_ROOT, "index")
DATA_KB_DIR = os.path.join(PROJECT_ROOT, "data_kb")
ANNUAL_REPORT_PATH = os.path.join(PROJECT_ROOT, "frontend", "satellite.html")
PARQUET_ROOT = os.path.join(PROJECT_ROOT, "data")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output")

# 卫星数据映射
BAG_LABEL = [
    '0x0824', '0x0822', '0x0823', '0x0826', '0x0A73', '0x0981', '0x0A6E', '0x0A6F',
    '0x0980', '0x0821', '0x08A3', '0x3030', '0x08A4', '0x09F5', '0x090E', '0x0A6D',
    '0x090F', '0x08A5', '0x090D', '0x08A6'
]
BAG_NAME = [
    '星务代码上注包', '星务中包', '星务慢包', 'DCS关键遥测包', 'VDES包', 'HWT905包', 'DCS慢包', 'ADSB包',
    '扩展板包', '星务快包', '主测控数传快包', '固定遥测包', '主测控数传慢包', 'GNSS包', '姿控中包', 'DCS快包',
    '姿控部组件包', '备测控数传快包', '姿控快包', '备测控数传慢包'
]

# OSS 配置
OSS_ACCESS_KEY_ID = os.environ.get('OSS_ACCESS_KEY_ID', 'YOUR_ACCESS_KEY_ID')
OSS_ACCESS_KEY_SECRET = os.environ.get('OSS_ACCESS_KEY_SECRET', 'YOUR_ACCESS_KEY_SECRET')
OSS_ENDPOINT = 'oss-cn-beijing.aliyuncs.com'
OSS_BUCKET_NAME = 'hede-satellite'

