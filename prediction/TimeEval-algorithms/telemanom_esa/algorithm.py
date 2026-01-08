import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

from telemanom.detector import Detector
from telemanom.modeling import Model
from telemanom.helpers import Config
from telemanom.channel import Channel
from typing import List
import argparse
import pandas as pd
import numpy as np
import json
import os
import sys
from dataclasses import dataclass, asdict, field
from dateutil.parser import parse as parse_date
from typing import List
from tensorflow.compat.v1 import set_random_seed


@dataclass
class CustomParameters:
    batch_size: int = 70
    smoothing_window_size: int = 30
    smoothing_perc: float = 0.05
    error_buffer: int = 100
    loss_metric: str = 'mse'
    optimizer: str = 'adam'
    split: float = 0.8
    validation_date_split: str = None
    dropout: float = 0.3
    lstm_batch_size: int = 64
    epochs: int = 35
    layers: List[int] = field(default_factory=lambda: [80, 80])
    early_stopping_patience: int = 10
    early_stopping_delta: float = 0.0003
    window_size: int = 250
    prediction_window_size: int = 10
    p: float = 0.13
    min_error_value: float = 0.05
    use_id: str = "internal-run-id"
    random_state: int = 42
    input_channels: List[str] = None
    input_channel_indices: List[int] = None  # do not use, automatically handled
    target_channels: List[str] = None
    target_channel_indices: List[int] = None  # do not use, automatically handled
    threshold_scores: bool = False


class AlgorithmArgs(argparse.Namespace):
    @property
    def ts(self) -> np.ndarray:
        print(f"Loading: {self.dataInput}")
        dataset, data_columns, anomaly_columns = self._read_dataset() # 读取用于训练的csv文件，整个读入！其中通道的数值类型为float，判断是否异常的标志为int8
        
        self._select_input_and_target_channels(data_columns) #根据参数筛选要用的通道，包括输入通道和输出通道，这里应该只有通道。

        all_used_channels = list(dict.fromkeys(self.customParameters.input_channels + self.customParameters.target_channels)) # 所有用的通道
        all_used_anomaly_columns = [f"is_anomaly_{ch}" for ch in all_used_channels] # 所有用来判断通道异常的列

        dataset = self._unravel_global_annotation(dataset, anomaly_columns, all_used_anomaly_columns) # 这里是兼容这种情况：标签里是is_anomaly，即只说明了异常，没说哪个通道异常，然后转化为每个通道都报异常，所以这里本来就符合，本质上没怎么执行
        dataset = dataset.loc[:, all_used_channels + all_used_anomaly_columns]

        data_columns = dataset.columns.tolist()[:len(all_used_channels)] # 这里是全部用到的列，应该不包括指令

        self._map_channels_to_indices(data_columns) # 这里是把列名映射成了下标（后续numpy通过下标访问）

        if self.executionType == "train":
            return self._prepare_training_data(dataset, all_used_anomaly_columns) #如果是训练模式的话，则准备数据并返回
        else:
            dataset = np.expand_dims(dataset.values[:, :-len(all_used_anomaly_columns)], axis=0).astype(np.float32)
            return dataset, None, None, None

    def _read_dataset(self):
        columns = pd.read_csv(self.dataInput, index_col="timestamp", nrows=0).columns.tolist()
        anomaly_columns = [x for x in columns if x.startswith("is_anomaly")]
        data_columns = columns[:-len(anomaly_columns)]

        dtypes = {col: np.float32 for col in data_columns}
        dtypes.update({col: np.uint8 for col in anomaly_columns})
        dataset = pd.read_csv(self.dataInput, index_col="timestamp", parse_dates=True, dtype=dtypes)

        return dataset, data_columns, anomaly_columns

    @staticmethod
    def get_valid_channels(raw_channels: List[str], data_cols: List[str], sort: bool = False) -> List[str]:
        if not raw_channels:
            print(f"No channels provided. Using all data columns: {data_cols}")
            valid_channels = data_cols
        else:
            valid_channels = list(dict.fromkeys([ch for ch in raw_channels if ch in data_cols]))
            if not valid_channels:
                print("No valid channels found in dataset, falling back to all data columns.")
                valid_channels = data_cols

        if sort:
            valid_channels.sort()

        return valid_channels

    def _select_input_and_target_channels(self, data_columns):
        self.customParameters.input_channels = self.get_valid_channels(
            self.customParameters.input_channels, data_columns, sort=True
        )
        self.customParameters.target_channels = self.get_valid_channels(
            self.customParameters.target_channels, data_columns, sort=True
        )

    @staticmethod
    def _unravel_global_annotation(dataset: pd.DataFrame, original_anomaly_cols: List[str],
                                   target_channel_anomaly_cols: List[str]) -> pd.DataFrame:
        if len(original_anomaly_cols) == 1 and original_anomaly_cols[0] == "is_anomaly":  # Handle datasets with only one global is_anomaly column
            for col in target_channel_anomaly_cols:
                dataset[col] = dataset["is_anomaly"]
            dataset = dataset.drop(columns="is_anomaly")
        return dataset

    def _map_channels_to_indices(self, data_columns):
        self.customParameters.input_channel_indices = [data_columns.index(x) for x in
                                                       self.customParameters.input_channels]
        self.customParameters.target_channel_indices = [data_columns.index(x) for x in
                                                        self.customParameters.target_channels]

    def _prepare_training_data(self, dataset, anomaly_columns):
        target_anomaly_column = "is_anomaly"
        dataset[target_anomaly_column] = 0

        for channel in self.customParameters.target_channels:
            dataset.loc[dataset[f"is_anomaly_{channel}"] > 0, f"is_anomaly_{channel}"] = 1
            dataset[target_anomaly_column] |= dataset[f"is_anomaly_{channel}"]

        dataset.drop(columns=anomaly_columns, inplace=True)

        labels_groups = dataset.groupby(
            (dataset[target_anomaly_column].shift() != dataset[target_anomaly_column]).cumsum())
        start_end_points = [
            (group[0], group[-1])
            for group in labels_groups.groups.values()
            if dataset.loc[group[0], target_anomaly_column] == 0
        ]
        dataset.drop(columns=[target_anomaly_column], inplace=True)

        binary_channels_mask = [
            np.sum(dataset.values[..., i].astype(np.int64) - dataset.values[..., i]) == 0 and
            len(np.unique(dataset.values[..., i])) == 2
            for i in range(dataset.values.shape[-1])
        ]
        channels_minimums = np.min(dataset.values, axis=0)
        channels_maximums = np.max(dataset.values, axis=0)

        validation_date_split = self._validate_date_split(dataset)

        if validation_date_split is None:
            dataset = np.array([dataset.loc[start:end].values for start, end in start_end_points], dtype=object)
        else:
            train_data, val_data = [], []
            for start_date, end_date in start_end_points:
                if end_date < validation_date_split:
                    train_data.append(dataset[start_date:end_date].values)
                elif start_date > validation_date_split:
                    val_data.append(dataset[start_date:end_date].values)
                else:
                    train_data.append(dataset[start_date:validation_date_split].values)
                    val_data.append(dataset[validation_date_split:end_date].values)
            dataset = [np.array(train_data), np.array(val_data)]

        return dataset, binary_channels_mask, channels_minimums, channels_maximums

    def _validate_date_split(self, dataset):
        validation_date_split = self.customParameters.validation_date_split
        if validation_date_split is not None:
            try:
                validation_date_split = parse_date(validation_date_split)
                if validation_date_split < dataset.index[0] or validation_date_split > dataset.index[-1]:
                    print(
                        f"Cannot use validation_date_split '{validation_date_split}' because it is outside the data range")
                    return None
            except:
                print(f"Cannot use validation_date_split '{validation_date_split}' because timestamp is not datetime")
                return None
        return validation_date_split


    @staticmethod
    def from_sys_args() -> 'AlgorithmArgs':
        args: dict = json.loads(sys.argv[1]) # 程序入口的第一步，读入输入    
        custom_parameter_keys = dir(CustomParameters())
        filtered_parameters = dict(filter(lambda x: x[0] in custom_parameter_keys, args.get("customParameters", {}).items()))
        hyper_params_path = os.path.join(os.path.dirname(args["dataOutput"]), "hyper_params.json")
        if os.path.isfile(hyper_params_path):
            with open(hyper_params_path, "r") as fh:
                hyper_params = json.load(fh)
            for key, value in hyper_params.items():
                filtered_parameters[key] = value
        args["customParameters"] = CustomParameters(**filtered_parameters)
        return AlgorithmArgs(**args)

def adapt_config_yaml(args: AlgorithmArgs) -> Config:
    params = asdict(args.customParameters)
    # remap config keys
    params["validation_split"] = 1 - params["split"]
    params["patience"] = params["early_stopping_patience"]
    params["min_delta"] = params["early_stopping_delta"]
    params["l_s"] = params["window_size"]
    for k in ["split", "early_stopping_patience", "early_stopping_delta"]:
        del params[k]

    params["meansOutput"] = f"{args.modelOutput}.means"
    params["stdsOutput"] = f"{args.modelOutput}.stds"

    config = Config.from_dict(params) # 这里应该只是参数名字的映射，ESA->NASA
    if args.executionType == "train":
        config["train"] = True
        config["predict"] = False
    elif args.executionType == "execute":
        config["train"] = False
        config["predict"] = True

    return config


def train(args: AlgorithmArgs, config: Config, channel: Channel):
    Model(config, config.use_id, channel, model_path=args.modelOutput)  # trains and saves model


def execute(args: AlgorithmArgs, config: Config, channels: Channel, thresholded: bool = False):
    detector = Detector(config=config, model_path=args.modelInput, result_path=args.dataOutput)
    errors = detector.predict(channels, args, thresholded)
    np.savetxt(args.dataOutput, errors, delimiter=",")


def set_random_state(config: AlgorithmArgs) -> None:
    seed = config.customParameters.random_state
    import random
    random.seed(seed)
    np.random.seed(seed)
    set_random_seed(seed)


def main():
    args = AlgorithmArgs.from_sys_args()
    set_random_state(args)

    ts, binary_channels_mask, channels_minimums, channels_maximums = args.ts #这里ts就是用于训练的数据集

    config = adapt_config_yaml(args) # 把命令行参数解析成最终的配置对象（config）
    is_train = args.executionType == "train"

    channels = Channel(config) # 根据 config 读取原始数据，并且把数据处理成模型能够训练的格式。
    channels.shape_data(ts, binary_channels_mask, channels_minimums, channels_maximums, train=is_train)
    '''
    # 这里应该制定好了数据生成器
    print("==========================")
    print("look here")
    print("==========================")
    print(args)
    while True:
        X, y = channels.generator_train[0]
        print("X shape:", X.shape)
        print("y shape:", y.shape)
        print(X[0])     # 打印第一个样本的数据
        print(y[0])     # 打印第一个样本的标签
    '''
    if is_train:
        train(args, config, channels)
    else:
        execute(args, config, channels, args.customParameters.threshold_scores)


if __name__ == "__main__":
    main()
