# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score

# 设置 matplotlib 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# ==========================================
# 1. LSTM 异常检测模型定义
# ==========================================

class TimeSeries(Dataset):
    """时序数据集封装"""
    def __init__(self, X, window_length: int, prediction_length: int, output_dims: Optional[List[int]] = None):
        self.output_dims = output_dims or list(range(X.shape[1]))
        self.X = torch.from_numpy(X).float()
        self.window_length = window_length
        self.prediction_length = prediction_length

    def __len__(self):
        return self.X.shape[0] - (self.window_length - 1) - self.prediction_length

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        end_idx = index + self.window_length
        x = self.X[index:end_idx]
        y = self.X[end_idx:end_idx+self.prediction_length, self.output_dims]
        return x, y.flatten()

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience: int, delta: float, epochs: int):
        self.patience = patience
        self.delta = delta
        self.epochs = epochs
        self.epochs_without_change = 0
        self.best_loss = float('inf')
        self.should_stop = False

    def update(self, loss: float):
        if loss < self.best_loss * (1 - self.delta):
            self.best_loss = loss
            self.epochs_without_change = 0
        else:
            self.epochs_without_change += 1
        
        if self.epochs_without_change >= self.patience:
            self.should_stop = True

class AnomalyScorer:
    """异常评分计算器 (基于马氏距离思想)"""
    def __init__(self):
        self.mean = torch.tensor(0, dtype=torch.float64)
        self.var = torch.tensor(1, dtype=torch.float64)

    def forward(self, errors: torch.Tensor) -> torch.Tensor:
        mean_diff = errors - self.mean
        return torch.mul(torch.mul(mean_diff, self.var**-1), mean_diff)

    def find_distribution(self, errors: torch.Tensor):
        self.mean = errors.mean(dim=0)
        self.var = errors.var(dim=0)
        self.var[self.var < 1e-6] = 1e-6

class LSTMAD(nn.Module):
    """LSTM 预测模型"""
    def __init__(self, input_size, lstm_layers=1, window_size=30, prediction_window_size=1, 
                 hidden_units=None, device="cpu"):
        super().__init__()
        self.hidden_units = hidden_units if hidden_units else input_size
        self.device = device
        
        self.lstms = nn.LSTM(input_size=input_size, 
                             hidden_size=self.hidden_units * prediction_window_size, 
                             batch_first=True, 
                             num_layers=lstm_layers).to(device)
        
        self.dense = nn.Linear(in_features=window_size * self.hidden_units * prediction_window_size, 
                               out_features=self.hidden_units * prediction_window_size).to(device)
        
        self.anomaly_scorer = AnomalyScorer()

    def forward(self, x):
        x, _ = self.lstms(x)
        x = x.reshape(x.size(0), -1)
        x = self.dense(x)
        return x

def train_model(model, train_data, val_data, config):
    """模型训练流程"""
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(config['patience'], config['delta'], config['epochs'])
    
    train_dataset = TimeSeries(train_data, config['window_size'], config['prediction_window_size'])
    val_dataset = TimeSeries(val_data, config['window_size'], config['prediction_window_size'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    print(f"开始训练: 训练集 {len(train_dataset)} 样本, 验证集 {len(val_dataset)} 样本")

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(model.device), y.to(model.device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(model.device), y.to(model.device)
                y_pred = model(x)
                val_loss += criterion(y_pred, y).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{config['epochs']} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
        
        early_stopping.update(val_loss)
        if early_stopping.should_stop:
            print("触发早停。")
            break

    print("计算异常评分分布参数...")
    model.eval()
    all_errors = []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(model.device), y.to(model.device)
            y_pred = model(x)
            all_errors.append(torch.abs(y_pred - y).cpu())
    
    if all_errors:
        model.anomaly_scorer.find_distribution(torch.cat(all_errors, dim=0))

def predict(model, test_data, config):
    """模型预测"""
    model.eval()
    dataset = TimeSeries(test_data, config['window_size'], config['prediction_window_size'])
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
    
    scores = []
    predictions = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(model.device), y.to(model.device)
            y_pred = model(x)
            error = torch.abs(y_pred - y).cpu()
            score = model.anomaly_scorer.forward(error).mean(dim=1)
            scores.append(score.numpy())
            predictions.append(y_pred.cpu().numpy())
            
    return np.concatenate(scores), np.concatenate(predictions)

# ==========================================
# 2. 绘图函数
# ==========================================

def plot_results(csv_path, output_image, target_points=5000):
    """绘制预测对比图"""
    if not os.path.exists(csv_path):
        return f"错误: 文件不存在 {csv_path}"

    print(f"正在绘图: {csv_path}...")
    df = pd.read_csv(csv_path)
    
    actual_cols = [c for c in df.columns if c.startswith('actual_')]
    pred_cols = [c for c in df.columns if c.startswith('pred_')]
    
    if not actual_cols:
        return "错误: CSV 中未找到 actual_ 开头的列"

    time_col = pd.to_datetime(df['timestamp']) if 'timestamp' in df.columns else df.index
    
    # 降采样以提高绘图速度
    stride = max(1, len(df) // target_points)
    df_sampled = df.iloc[::stride]
    time_sampled = time_col[::stride]

    num_features = len(actual_cols)
    fig, axes = plt.subplots(num_features, 1, figsize=(15, 5 * num_features), sharex=True)
    if num_features == 1: axes = [axes]

    for i, (act_col, pred_col) in enumerate(zip(actual_cols, pred_cols)):
        ax = axes[i]
        ax.plot(time_sampled, df_sampled[act_col], label='实际值', color='blue', alpha=0.6, linewidth=1)
        ax.plot(time_sampled, df_sampled[pred_col], label='预测值', color='orange', alpha=0.7, linewidth=1, linestyle='--')
        
        if 'label' in df.columns:
            anomalies = df_sampled[df_sampled['label'] == 1]
            if not anomalies.empty:
                ax.scatter(anomalies['timestamp'] if 'timestamp' in df.columns else anomalies.index, 
                           anomalies[act_col], color='red', label='异常标签', zorder=5, s=10)

        ax.set_title(f"通道: {act_col.replace('actual_', '')}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.xlabel("时间")
    plt.tight_layout()
    plt.savefig(output_image)
    plt.close()
    return output_image

# ==========================================
# 3. 主入口函数
# ==========================================

def run_lstm_detection_and_plot():
    """执行 LSTM 异常检测并绘图"""
    config = {
        'window_size': 100,
        'prediction_window_size': 1,  
        'batch_size': 512,
        'epochs': 3,
        'learning_rate': 0.001,
        'patience': 3,
        'delta': 0.0003,              
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # 修改路径：从 tools 目录向上两级找到根目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "prediction", "data", "preprocessed", "multivariate", "ESA-Mission1-semi-supervised")
    train_file = os.path.join(data_dir, "3_months.train.csv")
    test_file = os.path.join(data_dir, "84_months.test.csv")
    
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    result_csv = os.path.join(output_dir, "lstm_ad_results.csv")
    result_plot = os.path.join(output_dir, "lstm_prediction_plot.png")

    if not os.path.exists(train_file):
        return f"错误: 训练文件不存在 {train_file}"

    print(f"加载数据: {data_dir}...")
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    # 特征选择
    subset_channels = ["channel_41"]
    selected_features = [c for c in subset_channels if c in df_train.columns]
    
    # 数据预处理
    split_date = "2000-03-11"
    df_train['timestamp'] = pd.to_datetime(df_train['timestamp'])
    
    train_mask = df_train['timestamp'] < split_date
    val_mask = df_train['timestamp'] >= split_date
    
    X_train = df_train.loc[train_mask, selected_features].values
    X_val = df_train.loc[val_mask, selected_features].values
    
    # 缩减测试集以加快演示速度
    df_test = df_test.iloc[:len(df_test)//7]
    X_test = df_test[selected_features].values
    
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # 训练
    model = LSTMAD(input_size=len(selected_features), 
                   window_size=config['window_size'], 
                   prediction_window_size=config['prediction_window_size'],
                   device=config['device'])
    
    train_model(model, X_train, X_val, config)
    
    # 预测
    raw_scores, raw_predictions = predict(model, X_test, config)
    
    # 结果对齐
    pad_width = len(X_test) - len(raw_scores)
    final_scores = np.concatenate([np.zeros(pad_width), raw_scores])
    
    predictions_original = scaler.inverse_transform(raw_predictions)
    padding_preds = np.full((pad_width, predictions_original.shape[1]), np.nan)
    final_predictions = np.concatenate([padding_preds, predictions_original])
    
    # 保存结果
    result_df = pd.DataFrame({'timestamp': df_test['timestamp'], 'score': final_scores})
    
    for idx, col in enumerate(selected_features):
        result_df[f'actual_{col}'] = df_test[col]
        result_df[f'pred_{col}'] = final_predictions[:, idx]

    result_df.to_csv(result_csv, index=False)
    print(f"结果已保存: {result_csv}")
    
    return plot_results(result_csv, result_plot)
