
import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from typing import List, Optional, Tuple
import logging
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

# ==========================================
# 1. 移植 LSTM-AD 核心代码 (Dataset, Model, EarlyStopping)
# ==========================================

class TimeSeries(Dataset):
    def __init__(self, X, window_length: int, prediction_length: int, output_dims: Optional[List[int]] = None):
        self.output_dims = output_dims or list(range(X.shape[1]))
        self.X = torch.from_numpy(X).float()
        self.window_length = window_length
        self.prediction_length = prediction_length

    def __len__(self):
        return self.X.shape[0] - (self.window_length - 1) - self.prediction_length

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        end_idx = index+self.window_length
        x = self.X[index:end_idx]
        y = self.X[end_idx:end_idx+self.prediction_length, self.output_dims]
        return x, y.flatten() # Flatten y to match model output

class EarlyStopping:
    def __init__(self, patience: int, delta: float, epochs: int):
        self.patience = patience
        self.delta = delta
        self.epochs = epochs
        self.current_epoch = 0
        self.epochs_without_change = 0
        self.last_loss: Optional[float] = None
        self.best_loss = float('inf')
        self.should_stop = False

    def update(self, loss: float):
        self.current_epoch += 1
        if self.last_loss is None:
            self.last_loss = loss
            self.best_loss = loss
            return

        # Check if loss improved significantly
        if loss < self.best_loss * (1 - self.delta):
            self.best_loss = loss
            self.epochs_without_change = 0
        else:
            self.epochs_without_change += 1

        self.last_loss = loss
        if self.epochs_without_change >= self.patience:
            self.should_stop = True

class AnomalyScorer:
    def __init__(self):
        self.mean = torch.tensor(0, dtype=torch.float64)
        self.var = torch.tensor(1, dtype=torch.float64)

    def forward(self, errors: torch.Tensor) -> torch.Tensor:
        # Mahalanobis-like distance (simplified diagonal covariance)
        mean_diff = errors - self.mean
        return torch.mul(torch.mul(mean_diff, self.var**-1), mean_diff)

    def find_distribution(self, errors: torch.Tensor):
        self.mean = errors.mean(dim=0)
        self.var = errors.var(dim=0)
        # Avoid division by zero
        self.var[self.var < 1e-6] = 1e-6

class LSTMAD(nn.Module):
    def __init__(self, input_size, lstm_layers=1, window_size=30, prediction_window_size=1, 
                 hidden_units=None, device="cpu"):
        super().__init__()
        self.input_size = input_size
        self.window_size = window_size
        self.prediction_length = prediction_window_size
        self.hidden_units = hidden_units if hidden_units else input_size
        self.device = device

        self.lstms = nn.LSTM(input_size=input_size, 
                             hidden_size=self.hidden_units * self.prediction_length, 
                             batch_first=True, 
                             num_layers=lstm_layers).to(device)
        
        self.dense = nn.Linear(in_features=self.window_size * self.hidden_units * self.prediction_length, 
                               out_features=self.hidden_units * self.prediction_length).to(device)
        
        self.anomaly_scorer = AnomalyScorer()

    def forward(self, x):
        x, _ = self.lstms(x)
        x = x.reshape(x.size(0), -1) # Flatten: (batch, window * hidden * pred)
        x = self.dense(x)
        return x

# ==========================================
# 2. 训练与预测逻辑
# ==========================================

def train_model(model, train_data, val_data, config):
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(config['patience'], config['delta'], config['epochs'])
    
    train_dataset = TimeSeries(train_data, config['window_size'], config['prediction_window_size'])
    val_dataset = TimeSeries(val_data, config['window_size'], config['prediction_window_size'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples...")

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
                loss = criterion(y_pred, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{config['epochs']} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
        
        early_stopping.update(val_loss)
        if early_stopping.should_stop:
            print("Early stopping triggered.")
            break

    # Fit anomaly scorer distribution on validation data errors
    print("Fitting anomaly scorer distribution...")
    model.eval()
    all_errors = []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(model.device), y.to(model.device)
            y_pred = model(x)
            error = torch.abs(y_pred - y) # Absolute error
            all_errors.append(error.cpu())
    
    if all_errors:
        all_errors = torch.cat(all_errors, dim=0)
        model.anomaly_scorer.find_distribution(all_errors)

def predict(model, test_data, config):
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
            
            # Calculate anomaly score using the fitted scorer
            score = model.anomaly_scorer.forward(error)
            # Aggregate score (e.g., mean across dimensions)
            score = score.mean(dim=1)
            scores.append(score.numpy())
            predictions.append(y_pred.cpu().numpy())
            
    return np.concatenate(scores), np.concatenate(predictions)

# ==========================================
# 3. 主流程
# ==========================================

def main():
    # 配置 (Modified for speed)
    config = {
        'window_size': 100,           # Reduced for speed (was 250)
        'prediction_window_size': 1,  
        'batch_size': 512,            # Increased for speed (was 70)
        'epochs': 3,                 # Reduced for speed (was 1000)
        'learning_rate': 0.001,
        'patience': 3,                # Reduced for speed (was 20)
        'delta': 0.0003,              
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # 路径 (根据你的环境调整)
    data_dir = "data/preprocessed/multivariate/ESA-Mission1-semi-supervised"
    train_file = os.path.join(data_dir, "3_months.train.csv")
    test_file = os.path.join(data_dir, "84_months.test.csv")
    
    if not os.path.exists(train_file):
        print(f"Error: Train file not found at {train_file}")
        return

    print(f"Loading data from {data_dir}...")
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    # 1. Feature Selection (Strict Alignment)
    # mission1_experiments.py uses subset_channels for input_channels
    subset_channels = ["channel_41"]
    
    # Ensure these columns exist
    available_cols = df_train.columns
    selected_features = [c for c in subset_channels if c in available_cols]
    
    if len(selected_features) != len(subset_channels):
        print(f"Warning: Some subset channels not found. Using: {selected_features}")
    
    label_cols = [c for c in df_test.columns if c.startswith('is_anomaly_')]
    
    print(f"Selected Features (Strict): {selected_features}")
    
    # Filter DataFrames
    # We need timestamp for splitting, features for training, labels for eval
    # Keep timestamp for now
    
    # 2. Validation Split (Strict Alignment)
    # mission1_experiments.py: "3_months": "2000-03-11"
    split_date = "2000-03-11"
    
    # Convert timestamp to datetime
    df_train['timestamp'] = pd.to_datetime(df_train['timestamp'])
    
    train_mask = df_train['timestamp'] < split_date
    val_mask = df_train['timestamp'] >= split_date
    
    X_train_df = df_train.loc[train_mask, selected_features]
    X_val_df = df_train.loc[val_mask, selected_features]
    
    print(f"Training samples: {len(X_train_df)} (before {split_date})")
    print(f"Validation samples: {len(X_val_df)} (after {split_date})")

    # Reduce test set size to approx 6 months (Total is 84 months, so 6/84 ~= 1/14)
    print("Reducing test dataset size to approx 6 months for speed...")
    df_test = df_test.iloc[:len(df_test)//7]
    print(f"Reduced Test samples: {len(df_test)}")
    
    X_train = X_train_df.values
    X_val = X_val_df.values
    X_test = df_test[selected_features].values
    
    # 归一化
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val) # Use same scaler
    X_test = scaler.transform(X_test)
    
    # 初始化模型
    model = LSTMAD(input_size=len(selected_features), 
                   window_size=config['window_size'], 
                   prediction_window_size=config['prediction_window_size'],
                   device=config['device'])
    
    # 训练
    train_model(model, X_train, X_val, config)
    
    # 预测
    print("Predicting on test set...")
    raw_scores, raw_predictions = predict(model, X_test, config)
    
    # 对齐分数 (因为窗口滑动，前 window_size 个点没有分数)
    # 简单的填充策略：前面填 0
    pad_width = len(X_test) - len(raw_scores)
    padding = np.zeros(pad_width)
    final_scores = np.concatenate([padding, raw_scores])
    
    # 处理预测值
    predictions_original = scaler.inverse_transform(raw_predictions)
    padding_preds = np.full((pad_width, predictions_original.shape[1]), np.nan)
    final_predictions = np.concatenate([padding_preds, predictions_original])
    
    # 评估 (如果有标签)
    if label_cols:
        # 只要任意维度是异常，就认为是异常
        y_true = df_test[label_cols].max(axis=1).values
        # 简单的二值化
        y_true = (y_true > 0).astype(int)
        
        # 计算 AUC
        try:
            auc = roc_auc_score(y_true, final_scores)
            print(f"\nTest AUC: {auc:.4f}")
        except Exception as e:
            print(f"Could not calculate AUC: {e}")
            
        # 保存结果
        result_df = pd.DataFrame({
            'timestamp': df_test['timestamp'],
            'score': final_scores,
            'label': y_true
        })
        
        # Add actual and predicted values
        for idx, col_name in enumerate(selected_features):
            result_df[f'actual_{col_name}'] = df_test[col_name]
            result_df[f'pred_{col_name}'] = final_predictions[:, idx]

        result_df.to_csv("lstm_ad_results.csv", index=False)
        print("Results saved to lstm_ad_results.csv")
    else:
        print("No labels found in test set. Skipping evaluation.")

if __name__ == "__main__":
    main()
