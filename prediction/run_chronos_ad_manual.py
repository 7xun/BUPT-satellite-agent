import sys
import os

# Set Hugging Face Mirror
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for Chronos
try:
    from chronos import ChronosPipeline
except ImportError:
    logger.error("Chronos package not found.")
    logger.error("Please install it using: pip install git+https://github.com/amazon-science/chronos-forecasting.git")
    sys.exit(1)

# ==========================================
# 1. Anomaly Scorer (Aligned with LSTM-AD)
# ==========================================
class AnomalyScorer:
    def __init__(self):
        self.mean = torch.tensor(0, dtype=torch.float64)
        self.var = torch.tensor(1, dtype=torch.float64)

    def forward(self, errors: torch.Tensor) -> torch.Tensor:
        # Mahalanobis-like distance (simplified diagonal covariance)
        # errors shape: (batch, n_features)
        # mean shape: (n_features,)
        # var shape: (n_features,)
        
        # Ensure devices match
        if errors.device != self.mean.device:
            self.mean = self.mean.to(errors.device)
            self.var = self.var.to(errors.device)
            
        mean_diff = errors - self.mean
        return torch.mul(torch.mul(mean_diff, self.var**-1), mean_diff)

    def find_distribution(self, errors: torch.Tensor):
        self.mean = errors.mean(dim=0)
        self.var = errors.var(dim=0)
        # Avoid division by zero
        self.var[self.var < 1e-6] = 1e-6
        logger.info(f"Fitted AnomalyScorer: Mean={self.mean.cpu().numpy()}, Var={self.var.cpu().numpy()}")

# ==========================================
# 2. Dataset & Inference
# ==========================================
class ChronosDataset(Dataset):
    def __init__(self, data: np.ndarray, context_length: int, prediction_length: int, stride: int = 1):
        """
        Dataset for sliding window inference.
        data: 1D numpy array (one feature)
        """
        self.data = torch.from_numpy(data).float()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.stride = stride
        
    def __len__(self):
        total_len = self.data.shape[0]
        # We need context + prediction
        # Last valid start index such that start + context + prediction <= total_len
        max_start = total_len - self.context_length - self.prediction_length
        if max_start < 0:
            return 0
        return (max_start // self.stride) + 1

    def __getitem__(self, index):
        start_idx = index * self.stride
        context_end = start_idx + self.context_length
        target_end = context_end + self.prediction_length
        
        context = self.data[start_idx:context_end]
        target = self.data[context_end:target_end]
        
        return context, target

def run_inference_for_errors(pipeline, data_array, config, desc="Inference"):
    """
    Run Chronos inference on a multivariate data array (Time, Features).
    Returns: Errors (Time_windows, Features)
    """
    n_samples, n_features = data_array.shape
    
    # Store errors for each feature
    feature_errors = []
    
    for i in range(n_features):
        data_col = data_array[:, i]
        logger.info(f"Processing feature {i+1}/{n_features}...")
        
        dataset = ChronosDataset(
            data_col, 
            context_length=config['context_length'], 
            prediction_length=config['prediction_length'],
            stride=config['stride']
        )
        
        if len(dataset) == 0:
            logger.warning(f"Dataset empty for feature {i}. Data length {len(data_col)} < context {config['context_length']}")
            feature_errors.append(np.zeros((0,)))
            continue

        loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
        
        col_errors = []
        
        with torch.no_grad():
            for context, target in tqdm(loader, desc=f"{desc} Feat {i}", leave=False):
                # context: (batch, context_len)
                # target: (batch, pred_len)
                
                # Forecast
                forecast = pipeline.predict(
                    context, 
                    prediction_length=config['prediction_length'],
                    num_samples=20,
                    limit_prediction_length=False
                )
                # forecast shape: (batch_size, num_samples, prediction_length)
                
                # Get median forecast
                forecast_median = torch.quantile(forecast, 0.5, dim=1) # (batch_size, prediction_length)
                
                # Calculate error (MAE)
                target = target.to(forecast_median.device)
                error = torch.abs(forecast_median - target) # (batch_size, prediction_length)
                
                # Average error over prediction window if > 1
                error = error.mean(dim=1) # (batch_size,)
                
                col_errors.append(error.cpu())
        
        if col_errors:
            feature_errors.append(torch.cat(col_errors).numpy())
        else:
            feature_errors.append(np.array([]))
            
    # Stack errors: (n_windows, n_features)
    if not feature_errors or len(feature_errors[0]) == 0:
        return np.array([])
        
    return np.stack(feature_errors, axis=1)

# ==========================================
# 3. Main
# ==========================================
def main():
    # Configuration
    config = {
        'model_name': "amazon/chronos-t5-small", # 'tiny', 'small', 'base', 'large', '3b'
        'context_length': 1024,        # Chronos context length
        'prediction_length': 1,       # Predict 1 step ahead
        'batch_size': 128,             
        'stride': 1,                  
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'data_dir': "data/preprocessed/multivariate/ESA-Mission1-semi-supervised",
        'train_file': "3_months.train.csv",
        'test_file': "84_months.test.csv",
        'split_date': "2000-03-11"
    }
    
    train_path = os.path.join(config['data_dir'], config['train_file'])
    test_path = os.path.join(config['data_dir'], config['test_file'])
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        logger.error("Data files not found.")
        return

    # Load Model
    logger.info(f"Loading model: {config['model_name']} on {config['device']}")
    pipeline = ChronosPipeline.from_pretrained(
        config['model_name'],
        device_map=config['device'],
        torch_dtype=torch.bfloat16 if config['device'] == 'cuda' else torch.float32,
    )

    # Load Data
    logger.info("Loading data...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # Feature Selection (Strict Alignment)
    subset_channels = ["channel_41", "channel_42", "channel_43", "channel_44", "channel_45", "channel_46"]
    feature_cols = [c for c in subset_channels if c in df_train.columns]
    logger.info(f"Selected Features: {feature_cols}")
    
    # Split Train/Val
    df_train['timestamp'] = pd.to_datetime(df_train['timestamp'])
    val_mask = df_train['timestamp'] >= config['split_date']
    X_val_df = df_train.loc[val_mask, feature_cols]
    X_val = X_val_df.values
    
    X_test = df_test[feature_cols].values
    label_cols = [c for c in df_test.columns if c.startswith('is_anomaly_')]
    
    # Normalize (Fit on Train, but we only use Val for fitting scorer, so let's fit scaler on full train file for simplicity or just Val? 
    # LSTM script fits scaler on X_train (before split). Let's do the same for consistency.)
    # Actually LSTM script: X_train = df_train[...].values -> scaler.fit_transform(X_train) -> then split.
    # So scaler is fit on ALL of 3_months.train.csv.
    
    X_train_full = df_train[feature_cols].values
    scaler = MinMaxScaler()
    scaler.fit(X_train_full)
    
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Fit Anomaly Scorer on Validation Data
    logger.info(f"Running inference on Validation set ({len(X_val_scaled)} samples) to fit Scorer...")
    val_errors = run_inference_for_errors(pipeline, X_val_scaled, config, desc="Val")
    
    scorer = AnomalyScorer()
    if len(val_errors) > 0:
        val_errors_tensor = torch.from_numpy(val_errors).float().to(config['device'])
        scorer.find_distribution(val_errors_tensor)
    else:
        logger.error("No validation errors computed. Check data length vs context length.")
        return

    # 2. Run Inference on Test Data
    logger.info(f"Running inference on Test set ({len(X_test_scaled)} samples)...")
    test_errors = run_inference_for_errors(pipeline, X_test_scaled, config, desc="Test")
    
    if len(test_errors) == 0:
        logger.error("No test errors computed.")
        return
        
    # 3. Calculate Scores
    test_errors_tensor = torch.from_numpy(test_errors).float().to(config['device'])
    scores_tensor = scorer.forward(test_errors_tensor)
    
    # Aggregate scores (Mean across dimensions)
    final_scores = scores_tensor.mean(dim=1).cpu().numpy()
    
    # Align scores
    # We lost 'context_length' points at the beginning
    pad_width = config['context_length']
    padding = np.zeros(pad_width)
    full_scores = np.concatenate([padding, final_scores])
    
    # Truncate or Pad to match original length
    if len(full_scores) < len(df_test):
        full_scores = np.concatenate([full_scores, np.zeros(len(df_test) - len(full_scores))])
    elif len(full_scores) > len(df_test):
        full_scores = full_scores[:len(df_test)]
        
    # 4. Evaluation
    if label_cols:
        y_true = df_test[label_cols].max(axis=1).values
        y_true = (y_true > 0).astype(int)
        
        try:
            auc = roc_auc_score(y_true, full_scores)
            logger.info(f"Test AUC: {auc:.4f}")
        except Exception as e:
            logger.error(f"Could not calculate AUC: {e}")
            
        result_df = pd.DataFrame({
            'timestamp': df_test['timestamp'],
            'score': full_scores,
            'label': y_true
        })
        result_df.to_csv("chronos_ad_results.csv", index=False)
        logger.info("Results saved to chronos_ad_results.csv")

if __name__ == "__main__":
    main()
