
import argparse
import os
import statistics
from pathlib import Path
from glob import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from dateutil.parser import parse as parse_date
from timeeval import DatasetManager, Datasets
from timeeval.datasets import DatasetRecord
from utils import AnnotationLabel, encode_telecommands, find_full_time_range

# ================= 配置区域 =================
# 默认输入路径，如果命令行没传参数就用这个
DEFAULT_INPUT_PATH = "../../data/ESA-Mission"
# ===========================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path",
        type=str,
        nargs="?",
        default=DEFAULT_INPUT_PATH,
        help="Path to the folder with partial ESA Mission1 dataset.",
    )
    return parser.parse_args()

# 数据集切分时间点
dataset_splits = {
    "3_months": "2000-04-01",
    "10_months": "2000-11-01",
    "21_months": "2001-10-01",
    "42_months": "2003-07-01",
    "84_months": "2007-01-01"
}
test_data_split = "2007-01-01"

# 路径设置
args = parse_args()
data_raw_folder = args.input_path
current_dir = os.path.dirname(os.path.realpath(__file__))
data_processed_folder = os.path.abspath(os.path.join(current_dir, "../../data/preprocessed"))

dataset_collection_name = os.path.basename(data_raw_folder) # e.g., "ESA-Mission"
source_folder = Path(data_raw_folder)
target_folder = Path(data_processed_folder)

print(f"Source Data: {source_folder.absolute()}")
print(f"Output Data: {target_folder.absolute()}")

# 确保输出目录存在
dataset_type = "real"
input_type = "multivariate"
learning_type = "semi-supervised"
dataset_subfolder = Path(input_type) / f"{dataset_collection_name}-{learning_type}"
target_subfolder = target_folder / dataset_subfolder
target_subfolder.mkdir(parents=True, exist_ok=True)

def process_dataset(dm: DatasetManager, dataset_name: str, split_at: str, resampling_rule=pd.Timedelta(seconds=30)):
    print(f"\nProcessing dataset: {dataset_name} (Split at: {split_at})")
    
    # 1. 读取元数据 (Labels & Anomaly Types)
    # 即使是部分数据，通常也需要 labels.csv 来标记异常。如果缺失，我们将创建一个空的。
    try:
        labels_df = pd.read_csv(source_folder / "labels.csv", parse_dates=["StartTime", "EndTime"], date_parser=lambda x: parse_date(x, ignoretz=True))
        anomaly_types_df = pd.read_csv(source_folder / "anomaly_types.csv")
        print("  Loaded labels.csv and anomaly_types.csv")
    except FileNotFoundError:
        print("  WARNING: labels.csv or anomaly_types.csv not found. Generating unlabeled data.")
        labels_df = pd.DataFrame(columns=["Channel", "StartTime", "EndTime", "ID"])
        anomaly_types_df = pd.DataFrame(columns=["ID", "Category"])

    # 2. 扫描现有的 Channel 文件
    # 自动检测 data/ESA-Mission/channels 下有哪些 zip 文件
    extension = ".zip"
    channels_dir = source_folder / "channels"
    if not channels_dir.exists():
        print(f"  ERROR: Channels directory not found at {channels_dir}")
        return

    all_parameter_names = sorted([
        os.path.basename(f)[: -len(extension)]
        for f in glob(str(channels_dir / f"*{extension}"))
    ])
    
    if not all_parameter_names:
        print("  ERROR: No channel data (*.zip) found in channels directory!")
        return
    
    print(f"  Found {len(all_parameter_names)} channels: {all_parameter_names}")

    # 3. 处理 Telecommands (可选)
    # 如果为了节省空间没有下载 telecommands，这里会自动跳过
    try:
        telecommands_df = pd.read_csv(source_folder / "telecommands.csv")
        telecommands_min_priority = 3
        telecommands_df = telecommands_df.loc[telecommands_df["Priority"] >= telecommands_min_priority]
        all_telecommands_names = sorted(telecommands_df.Telecommand.to_list())
        print(f"  Found telecommands configuration, will look for {len(all_telecommands_names)} telecommand files.")
    except FileNotFoundError:
        print("  Telecommands.csv not found. Skipping telecommands processing.")
        telecommands_df = pd.DataFrame()
        all_telecommands_names = []

    # 准备文件路径
    train_test_paths = {"train": None, "test": None}
    target_meta_filepath = target_subfolder / f"{dataset_name}.{Datasets.METADATA_FILENAME_SUFFIX}"

    # 4. 生成 Train 和 Test 文件
    for train_test_type in ["train", "test"]:
        # Test set 总是使用 84_months 的数据范围
        train_test_name = "84_months" if train_test_type == "test" else dataset_name
        
        processed_filename = f"{train_test_name}.{train_test_type}.csv"
        target_filepath = target_subfolder / processed_filename
        train_test_paths[train_test_type] = str(dataset_subfolder / processed_filename).replace(os.sep, '/')

        # 如果文件已存在，跳过生成
        if target_filepath.exists() and target_meta_filepath.exists():
            print(f"  Skipping {processed_filename} (already exists)")
            continue

        print(f"  Generating {processed_filename}...")
        params_dict = {}

        # --- 处理 Channels ---
        for param in tqdm(all_parameter_names, desc="  Channels"):
            param_path = channels_dir / f"{param}{extension}"
            param_df = pd.read_pickle(param_path)
            param_df["label"] = np.uint8(0)
            param_df = param_df.rename(columns={param: "value"})

            # 特殊预处理：对单调通道做差分 (Channel 4-11)
            # 注意：这里假设 param 名字格式为 "channel_X"
            try:
                channel_id = int(param.split("_")[1])
                if 4 <= channel_id <= 11:
                    param_df.value = np.diff(param_df.value, append=param_df.value[-1])
            except (IndexError, ValueError):
                pass # 如果名字格式不对，就不做差分

            # 打标签
            is_param_annotated = False
            for _, row in labels_df.iterrows():
                if row["Channel"] == param:
                    matches = anomaly_types_df.loc[anomaly_types_df["ID"] == row["ID"]]["Category"].values
                    if len(matches) > 0:
                        atype = matches[0]
                        label_val = AnnotationLabel.ANOMALY.value if atype == "Anomaly" else (AnnotationLabel.RARE_EVENT.value if atype == "Rare Event" else AnnotationLabel.GAP.value)
                        param_df.loc[row["StartTime"]:row["EndTime"], "label"] = label_val
                        is_param_annotated = True

            # 切分数据 (Train vs Test)
            if split_at is not None:
                if train_test_type == "train":
                    param_df = param_df[param_df.index <= parse_date(split_at)].copy()
                else:
                    param_df = param_df[param_df.index > parse_date(test_data_split)].copy()

            if param_df.empty:
                params_dict[param] = pd.DataFrame() # 空占位
                continue

            # 重采样 (Resampling)
            first_idx = pd.Timestamp(param_df.index[0]).floor(freq=resampling_rule)
            last_idx = pd.Timestamp(param_df.index[-1]).ceil(freq=resampling_rule)
            resampled_range = pd.date_range(first_idx, last_idx, freq=resampling_rule)
            
            # 使用 ffill 重采样
            resampled_df = param_df.reindex(resampled_range, method="ffill")
            # 修复第一个点可能的 NaN
            if not param_df.empty:
                resampled_df.iloc[0] = param_df.iloc[0]
            
            params_dict[param] = resampled_df

            # 恢复因重采样丢失的短异常标签
            if is_param_annotated:
                grouper = param_df.groupby(pd.Grouper(freq=resampling_rule))
                for timestamp, group in grouper.indices.items():
                    if len(group) <= 1: continue
                    org_elements = param_df.iloc[group]
                    # 只有当重采样后的点是正常的，但原始数据里有异常时才恢复
                    # 这里简化逻辑：只要原始区间里有异常，就强制标记该时间点
                    is_annotated = (org_elements.label == AnnotationLabel.ANOMALY.value) | (org_elements.label == AnnotationLabel.RARE_EVENT.value)
                    if is_annotated.any():
                        target_idx = timestamp + pd.Timedelta(resampling_rule)
                        if target_idx in params_dict[param].index:
                            # 使用 .values 避免索引对齐错误
                            params_dict[param].loc[target_idx] = org_elements[is_annotated].iloc[-1].values

        # --- 处理 Telecommands (如果有) ---
        if all_telecommands_names:
            for param in tqdm(all_telecommands_names, desc="  Telecommands"):
                tc_path = source_folder / "telecommands" / f"{param}{extension}"
                if not tc_path.exists():
                    continue
                
                param_df = pd.read_pickle(tc_path)
                param_df["label"] = np.uint8(0)
                param_df = param_df.rename(columns={param: "value"})
                param_df.index = pd.to_datetime(param_df.index)
                param_df = param_df[~param_df.index.duplicated()]
                param_df = encode_telecommands(param_df, resampling_rule)

                if split_at is not None:
                    if train_test_type == "train":
                        param_df = param_df[param_df.index <= parse_date(split_at)].copy()
                    else:
                        param_df = param_df[param_df.index > parse_date(test_data_split)].copy()

                if len(param_df) == 0:
                    continue

                first_idx = pd.Timestamp(param_df.index[0]).floor(freq=resampling_rule)
                last_idx = pd.Timestamp(param_df.index[-1]).ceil(freq=resampling_rule)
                resampled_range = pd.date_range(first_idx, last_idx, freq=resampling_rule)
                resampled_df = param_df.reindex(resampled_range, method="ffill")
                resampled_df.iloc[0] = param_df.iloc[0]
                params_dict[param] = resampled_df

        # --- 合并所有列 ---
        if not params_dict:
            print("  No data found for this split. Skipping.")
            continue

        start_time, end_time = find_full_time_range(params_dict)
        full_index = pd.date_range(start_time, end_time, freq=resampling_rule)
        data_df = pd.DataFrame(index=full_index)

        for param in list(params_dict.keys()):
            df = params_dict.pop(param)
            if df.empty:
                continue
            
            df = df.rename(columns={"value": param, "label": f"is_anomaly_{param}"})
            # 合并到主表
            data_df = data_df.join(df, how='left')
            
            # 填充 NaN
            data_df[param] = data_df[param].ffill().bfill()
            data_df[f"is_anomaly_{param}"] = data_df[f"is_anomaly_{param}"].ffill().bfill().fillna(0).astype(np.uint8)

        # 整理列顺序
        cols = data_df.columns.tolist()
        # 简单的排序：先放值列，再放标签列
        value_cols = sorted([c for c in cols if not c.startswith("is_anomaly_")])
        label_cols = sorted([c for c in cols if c.startswith("is_anomaly_")])
        data_df = data_df[value_cols + label_cols]
        
        # 插入 timestamp
        data_df.insert(0, "timestamp", data_df.index.strftime('%Y-%m-%d %H:%M:%S'))

        # 保存 CSV
        data_df.to_csv(target_filepath, index=False, lineterminator='\n')
        print(f"  Saved {target_filepath}")

        # --- 注册元数据到 DatasetManager (仅在处理 Train 集时注册) ---
        if train_test_type == "train":
            # 需要读取对应的 Test 集来计算指标
            test_filepath = target_subfolder / "84_months.test.csv"
            if test_filepath.exists():
                df_test = pd.read_csv(test_filepath)
                
                # 计算污染率等统计信息
                anomaly_cols = [c for c in df_test.columns if c.startswith("is_anomaly_")]
                if anomaly_cols:
                    labels = df_test[anomaly_cols].max(axis=1).values
                    # Ensure labels are binary (0 or 1) for correct diff calculation
                    labels = (labels > 0).astype(int)
                    n_anomalies = np.sum(labels)
                    contamination = n_anomalies / len(df_test)
                    
                    # 计算异常长度
                    diffs = np.diff(labels, prepend=0, append=0)
                    lengths = np.where(diffs == -1)[0] - np.where(diffs == 1)[0]
                    if len(lengths) == 0:
                        min_l, med_l, max_l = 0, 0, 0
                    else:
                        min_l, med_l, max_l = lengths.min(), np.median(lengths), lengths.max()
                else:
                    contamination, n_anomalies, min_l, med_l, max_l = 0, 0, 0, 0, 0

                # 注册
                dm.add_dataset(DatasetRecord(
                    collection_name=dataset_collection_name,
                    dataset_name=dataset_name,
                    train_path=train_test_paths["train"],
                    test_path=train_test_paths["test"],
                    dataset_type=dataset_type,
                    datetime_index=True,
                    split_at=split_at,
                    train_type=learning_type,
                    train_is_normal=False,
                    input_type=input_type,
                    length=len(pd.read_csv(target_filepath)),
                    dimensions=len(value_cols),
                    contamination=contamination,
                    num_anomalies=n_anomalies,
                    min_anomaly_length=min_l,
                    median_anomaly_length=med_l,
                    max_anomaly_length=max_l,
                    mean=0, stddev=0, trend="no trend", stationarity="not_stationary", period_size=0
                ))
                print(f"  Registered metadata for {dataset_name}")

    dm.save()

if __name__ == "__main__":
    dm = DatasetManager(data_processed_folder)
    # 遍历所有预定义的切分点进行处理
    for name, split in dataset_splits.items():
        process_dataset(dm, name, split)
