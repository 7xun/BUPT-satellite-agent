import json
import sys
# 引入你的算法主入口
from algorithm import main

# 1. 构造 TimeEval 平时会自动生成的参数字典
# 注意：这里的路径要改成你本地电脑的真实路径！
train_params = {
    "executionType": "train",
    "dataInput": "data/preprocessed/multivariate/ESA-Mission1-semi-supervised/3_months.train.csv",        # 本地训练集路径
    "modelOutput": "TimeEval-algorithms/telemanom_esa/test_output/1",   # 模型保存路径
    "dataOutput": "TimeEval-algorithms/telemanom_esa/test_output/2",    # 训练阶段这个可以随便填
    "customParameters": {
        "epochs": 1,             # 调试时设小一点，跑通就行
        "batch_size": 32,
        "window_size": 250,
        # "input_channels": ["channel_41", "channel_42", "channel_43", "channel_44", "channel_45", "channel_46"], # 你的数据列名
        "target_channels": ["channel_41", "channel_42", "channel_43", "channel_44", "channel_45", "channel_46"]
    }
}

# 2. 模拟命令行参数传入
# TimeEval 是通过 sys.argv[1] 传入 JSON 字符串的
json_str = json.dumps(train_params)
sys.argv = ["algorithm.py", json_str]

# 3. 运行！
if __name__ == "__main__":
    main()