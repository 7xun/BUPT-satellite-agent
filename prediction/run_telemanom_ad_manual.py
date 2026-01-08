import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from typing import List, Optional, Tuple, Dict
import logging
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
import itertools

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# 1. LSTM Model (Same as LSTM-AD)
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
        return x, y.flatten()

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

        if loss < self.best_loss * (1 - self.delta):
            self.best_loss = loss
            self.epochs_without_change = 0
        else:
            self.epochs_without_change += 1

        self.last_loss = loss
        if self.epochs_without_change >= self.patience:
            self.should_stop = True

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
                               out_features=self.input_size * self.prediction_length).to(device)
        
    def forward(self, x):
        x, _ = self.lstms(x)
        x = x.reshape(x.size(0), -1)
        x = self.dense(x)
        return x

# ==========================================
# 2. Telemanom Logic (Errors, Thresholding)
# ==========================================

def consecutive_groups(iterable):
    """Helper to replace more_itertools.consecutive_groups"""
    s = sorted(set(iterable))
    for k, g in itertools.groupby(enumerate(s), lambda x: x[0] - x[1]):
        yield map(lambda x: x[1], g)

class TelemanomScorer:
    def __init__(self, config):
        self.config = config
        self.smoothing_window_size = config.get('smoothing_window_size', 30)
        self.smoothing_perc = config.get('smoothing_perc', 0.05)
        self.error_buffer = config.get('error_buffer', 100)
        self.p = config.get('p', 0.13) # Minimum percent decrease between anomalies
        self.min_error_value = config.get('min_error_value', 0.05)
        
    def process(self, y_true, y_pred):
        """
        Process predictions to generate anomaly scores.
        y_true: (n_samples, n_channels)
        y_pred: (n_samples, n_channels)
        """
        # 1. Calculate Errors
        e = np.abs(y_true - y_pred)
        
        # 2. Smoothing (EWMA)
        # Telemanom uses a dynamic window size based on batch size, but here we use global
        smoothing_window = int(self.config['batch_size'] * self.smoothing_window_size * self.smoothing_perc)
        smoothing_window = max(1, smoothing_window) # Safety
        
        e_s = pd.DataFrame(e).ewm(span=smoothing_window).mean().values
        
        # 3. Process each channel
        n_channels = e_s.shape[1]
        combined_scores = np.zeros(len(e_s))
        
        for i in range(n_channels):
            channel_scores = self._process_channel(e_s[:, i], y_true[:, i])
            # Telemanom usually aggregates by taking the max score across channels if looking for system anomaly
            # Or we can sum them. Here we take max.
            combined_scores = np.maximum(combined_scores, channel_scores)
            
        return combined_scores

    def _process_channel(self, e_s, y_test):
        """
        Process a single channel's smoothed errors to find anomalies.
        """
        # Basic stats
        mean_e_s = np.mean(e_s)
        sd_e_s = np.std(e_s)
        e_s_inv = np.array([mean_e_s + (mean_e_s - e) for e in e_s])
        
        sd_lim = 12.0
        epsilon = mean_e_s + sd_lim * sd_e_s
        epsilon_inv = mean_e_s + sd_lim * sd_e_s
        
        # Find best epsilon
        epsilon, sd_threshold = self._find_epsilon(e_s, mean_e_s, sd_e_s, sd_lim)
        epsilon_inv, sd_threshold_inv = self._find_epsilon(e_s_inv, mean_e_s, sd_e_s, sd_lim)
        
        # Compare to epsilon (Find candidate anomalies)
        i_anom = self._compare_to_epsilon(e_s, epsilon, y_test)
        i_anom_inv = self._compare_to_epsilon(e_s_inv, epsilon_inv, y_test)
        
        # Prune
        i_anom = self._prune_anoms(i_anom, e_s, epsilon)
        i_anom_inv = self._prune_anoms(i_anom_inv, e_s_inv, epsilon_inv)
        
        # Merge
        all_anom = np.sort(np.unique(np.concatenate([i_anom, i_anom_inv]))).astype(int)
        
        # Score
        scores = np.zeros(len(e_s))
        groups = [list(g) for g in consecutive_groups(all_anom)]
        
        for seq in groups:
            if not seq: continue
            start, end = seq[0], seq[-1]
            
            # Score calculation from Telemanom
            # max_score = max( (e_s - epsilon) / (mean + std) )
            
            # Regular score
            denom = mean_e_s + sd_e_s
            if denom == 0: denom = 1e-6
            
            score_reg = 0
            if len(i_anom) > 0:
                 # Check if this sequence overlaps with i_anom
                 if any(idx in i_anom for idx in seq):
                     score_reg = np.max(np.abs(e_s[start:end+1] - epsilon)) / denom
            
            score_inv = 0
            if len(i_anom_inv) > 0:
                if any(idx in i_anom_inv for idx in seq):
                    score_inv = np.max(np.abs(e_s_inv[start:end+1] - epsilon_inv)) / denom
            
            final_score = max(score_reg, score_inv)
            scores[start:end+1] = final_score
            
        return scores

    def _find_epsilon(self, e_s, mean_e_s, sd_e_s, sd_lim):
        if sd_e_s == 0: return mean_e_s, 0
        
        max_score = -np.inf
        best_epsilon = mean_e_s + sd_lim * sd_e_s
        best_z = sd_lim
        
        for z in np.arange(2.5, sd_lim, 0.5):
            epsilon = mean_e_s + (sd_e_s * z)
            pruned_e_s = e_s[e_s < epsilon]
            
            i_anom = np.argwhere(e_s >= epsilon).flatten()
            
            # Buffer
            if len(i_anom) > 0:
                # Expand indices by buffer (simplified)
                # In full Telemanom, they use a loop. Here we can use convolution or just skip for speed if needed.
                # Let's stick to simple expansion
                pass 
            
            if len(i_anom) > 0:
                groups = [list(g) for g in consecutive_groups(i_anom)]
                E_seq = [g for g in groups if len(g) > 0] # Telemanom checks g[0]!=g[-1], meaning len>1?
                # "if not g[0] == g[-1]" means length > 1. Single points are ignored?
                E_seq = [g for g in E_seq if g[0] != g[-1]]
                
                num_seq = len(E_seq)
                num_anom = len(i_anom)
                
                if num_seq == 0: continue

                mean_perc_decrease = (mean_e_s - np.mean(pruned_e_s)) / mean_e_s
                sd_perc_decrease = (sd_e_s - np.std(pruned_e_s)) / sd_e_s
                
                denom = (num_seq ** 2 + num_anom)
                if denom == 0: denom = 1
                
                score = (mean_perc_decrease + sd_perc_decrease) / denom
                
                if score >= max_score and num_seq <= 5 and num_anom < (len(e_s) * 0.5):
                    max_score = score
                    best_epsilon = epsilon
                    best_z = z
                    
        return best_epsilon, best_z

    def _compare_to_epsilon(self, e_s, epsilon, y_test):
        # Check minimum error value
        if np.max(e_s) < self.min_error_value:
            return np.array([])
            
        i_anom = np.argwhere(e_s >= epsilon).flatten()
        
        if len(i_anom) == 0: return np.array([])
        
        # Buffer logic (simplified: just expand)
        # Telemanom expands by error_buffer
        # We will skip complex buffer logic for brevity, but it helps connect close anomalies
        
        return i_anom

    def _prune_anoms(self, i_anom, e_s, epsilon):
        if len(i_anom) == 0: return i_anom
        
        groups = [list(g) for g in consecutive_groups(i_anom)]
        E_seq = [g for g in groups if len(g) > 1] # Only keep sequences > 1
        
        if not E_seq: return np.array([])
        
        # Calculate max error in each sequence
        seq_maxs = []
        for seq in E_seq:
            seq_maxs.append(np.max(e_s[seq]))
            
        # Sort by max error
        # ... (Pruning logic is complex, skipping for "Manual Script" simplicity unless requested)
        # The core value of Telemanom is the dynamic thresholding (find_epsilon), which we implemented.
        
        # Reconstruct i_anom from E_seq
        final_i_anom = []
        for seq in E_seq:
            final_i_anom.extend(seq)
            
        return np.array(final_i_anom)

# ==========================================
# 3. Training & Main
# ==========================================

def train_model(model, train_data, val_data, config):
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(config['patience'], config['delta'], config['epochs'])
    
    train_dataset = TimeSeries(train_data, config['window_size'], config['prediction_window_size'])
    val_dataset = TimeSeries(val_data, config['window_size'], config['prediction_window_size'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    logger.info(f"Training on {len(train_dataset)} samples...")

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

        if (epoch+1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{config['epochs']} - Train: {train_loss:.6f} - Val: {val_loss:.6f}")
        
        early_stopping.update(val_loss)
        if early_stopping.should_stop:
            logger.info("Early stopping triggered.")
            break

def predict_and_score(model, test_data, config):
    model.eval()
    dataset = TimeSeries(test_data, config['window_size'], config['prediction_window_size'])
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(model.device)
            y_pred = model(x)
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y.cpu().numpy()) # y is flattened in dataset
            
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    
    # Reshape back to (samples, channels)
    # The model output is flattened (batch, channels*pred_len). Pred_len=1.
    n_channels = config['n_channels']
    y_pred = y_pred.reshape(-1, n_channels)
    y_true = y_true.reshape(-1, n_channels)
    
    # Telemanom Scoring
    scorer = TelemanomScorer(config)
    scores = scorer.process(y_true, y_pred)
    
    return scores

def main():
    # Configuration (Telemanom ESA defaults)
    config = {
        'window_size': 250,
        'prediction_window_size': 1,
        'batch_size': 70,
        'epochs': 1000,
        'learning_rate': 0.001,
        'patience': 20,
        'delta': 0.0003,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # Telemanom specific
        'smoothing_window_size': 30,
        'smoothing_perc': 0.05,
        'error_buffer': 100,
        'p': 0.13,
        'min_error_value': 0.05
    }
    
    data_dir = "data/preprocessed/multivariate/ESA-Mission1-semi-supervised"
    train_file = os.path.join(data_dir, "3_months.train.csv")
    test_file = os.path.join(data_dir, "84_months.test.csv")
    
    if not os.path.exists(train_file):
        logger.error(f"Train file not found: {train_file}")
        return

    logger.info("Loading data...")
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    # Feature Selection (Subset)
    subset_channels = ["channel_41", "channel_42", "channel_43", "channel_44", "channel_45", "channel_46"]
    feature_cols = [c for c in subset_channels if c in df_train.columns]
    config['n_channels'] = len(feature_cols)
    
    logger.info(f"Selected Features: {feature_cols}")
    
    # Split
    split_date = "2000-03-11"
    df_train['timestamp'] = pd.to_datetime(df_train['timestamp'])
    
    train_mask = df_train['timestamp'] < split_date
    val_mask = df_train['timestamp'] >= split_date
    
    X_train = df_train.loc[train_mask, feature_cols].values
    X_val = df_train.loc[val_mask, feature_cols].values
    X_test = df_test[feature_cols].values
    
    # Normalize
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Model
    model = LSTMAD(input_size=len(feature_cols), 
                   lstm_layers=2, # Telemanom uses [80, 80] -> 2 layers
                   hidden_units=80,
                   window_size=config['window_size'], 
                   prediction_window_size=config['prediction_window_size'],
                   device=config['device'])
    
    # Train
    train_model(model, X_train, X_val, config)
    
    # Predict & Score
    logger.info("Predicting and scoring...")
    raw_scores = predict_and_score(model, X_test, config)
    
    # Pad scores
    padding = np.zeros(len(X_test) - len(raw_scores))
    final_scores = np.concatenate([padding, raw_scores])
    
    # Evaluate
    label_cols = [c for c in df_test.columns if c.startswith('is_anomaly_')]
    if label_cols:
        y_true = df_test[label_cols].max(axis=1).values
        y_true = (y_true > 0).astype(int)
        
        try:
            auc = roc_auc_score(y_true, final_scores)
            logger.info(f"Test AUC: {auc:.4f}")
        except Exception as e:
            logger.error(f"AUC Error: {e}")
            
        result_df = pd.DataFrame({
            'timestamp': df_test['timestamp'],
            'score': final_scores,
            'label': y_true
        })
        result_df.to_csv("telemanom_esa_results.csv", index=False)
        logger.info("Results saved to telemanom_esa_results.csv")

if __name__ == "__main__":
    main()
