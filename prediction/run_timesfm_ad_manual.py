import sys
import os

# Set Hugging Face Mirror (Optional, consistent with Chronos script)
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

# Check for TimesFM
try:
    import timesfm
except ImportError:
    logger.error("TimesFM package not found.")
    logger.error("Please install it using: pip install timesfm")
    sys.exit(1)

# ==========================================
# 1. Anomaly Scorer (Same as Chronos/LSTM-AD)
# ==========================================
class AnomalyScorer:
    def __init__(self):
        self.mean = torch.tensor(0, dtype=torch.float64)
        self.var = torch.tensor(1, dtype=torch.float64)

    def forward(self, errors: torch.Tensor) -> torch.Tensor:
        # Mahalanobis-like distance
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
class TimesFMDataset(Dataset):
    def __init__(self, data: np.ndarray, context_length: int, prediction_length: int, stride: int = 1):
        """
        Dataset for sliding window inference.
        data: 1D numpy array (one feature)
        """
        self.data = data # Keep as numpy for TimesFM which often expects numpy inputs
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.stride = stride
        
    def __len__(self):
        total_len = self.data.shape[0]
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

def run_inference_for_errors(tfm_model, data_array, config, desc="Inference"):
    """
    Run TimesFM inference on a multivariate data array.
    """
    n_samples, n_features = data_array.shape
    feature_errors = []
    feature_forecasts = []
    
    for i in range(n_features):
        data_col = data_array[:, i]
        logger.info(f"Processing feature {i+1}/{n_features}...")
        
        dataset = TimesFMDataset(
            data_col, 
            context_length=config['context_length'], 
            prediction_length=config['prediction_length'],
            stride=config['stride']
        )
        
        if len(dataset) == 0:
            logger.warning(f"Dataset empty for feature {i}.")
            feature_errors.append(np.zeros((0,)))
            feature_forecasts.append(np.zeros((0,)))
            continue

        # TimesFM works well with batching, but inputs are usually lists of arrays
        loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
        
        col_errors = []
        col_forecasts = []
        
        # No torch.no_grad() needed here strictly as TimesFM handles it, but good practice if wrapping
        for context_batch, target_batch in tqdm(loader, desc=f"{desc} Feat {i}", leave=False):
            # context_batch: (batch, context_len) - Tensor from DataLoader
            # target_batch: (batch, pred_len)
            
            # Convert to list of numpy arrays for TimesFM
            context_list = [c.numpy() for c in context_batch]
            
            # Forecast
            # freq=[0] implies high frequency / unknown, suitable for sensor data
            # forecast shape: (batch, horizon) for point forecast
            point_forecast, _ = tfm_model.forecast(context_list, freq=[0]*len(context_list))
            
            # Convert back to tensor for error calc
            forecast_tensor = torch.from_numpy(point_forecast).float()
            target_tensor = target_batch.float()
            
            # Calculate Error (MAE)
            # Ensure shapes match: (batch, pred_len)
            if forecast_tensor.shape != target_tensor.shape:
                # Sometimes TimesFM might return (batch, horizon, 1) or similar
                forecast_tensor = forecast_tensor.view(target_tensor.shape)

            error = torch.abs(forecast_tensor - target_tensor)
            error = error.mean(dim=1) # Average over horizon
            
            col_errors.append(error)
            col_forecasts.append(forecast_tensor.mean(dim=1))
        
        if col_errors:
            feature_errors.append(torch.cat(col_errors).numpy())
            feature_forecasts.append(torch.cat(col_forecasts).numpy())
        else:
            feature_errors.append(np.array([]))
            feature_forecasts.append(np.array([]))
            
    if not feature_errors or len(feature_errors[0]) == 0:
        return np.array([]), np.array([])
        
    return np.stack(feature_errors, axis=1), np.stack(feature_forecasts, axis=1)

# ==========================================
# 3. Main
# ==========================================
def main():
    # Configuration
    config = {
        'model_name': "google/timesfm-1.0-200m-pytorch", 
        'context_length': 1024,        # TimesFM standard context
        'prediction_length': 1,       # Predict 1 step ahead
        'batch_size': 256,             
        'stride': 1,                  
        'device': "gpu" if torch.cuda.is_available() else "cpu", # TimesFM uses 'gpu'/'cpu' string
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
    
    # Initialize TimesFM
    # Note: input_patch_len=32, output_patch_len=128 are standard for the 200m model
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            context_len=config['context_length'],
            horizon_len=config['prediction_length'],
            input_patch_len=32,
            output_patch_len=128,
            num_layers=20,
            model_dims=1280,
            backend=config['device']
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id=config['model_name']
        )
    )
    
    # Load checkpoint (Already handled in __init__ if passed, but let's be safe or check if needed)
    # Based on signature, __init__ takes checkpoint, so it likely loads it.


    # Load Data
    logger.info("Loading data...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # Feature Selection
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
    
    # Normalize
    X_train_full = df_train[feature_cols].values
    scaler = MinMaxScaler()
    scaler.fit(X_train_full)
    
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Fit Anomaly Scorer on Validation Data
    logger.info(f"Running inference on Validation set ({len(X_val_scaled)} samples) to fit Scorer...")
    val_errors, _ = run_inference_for_errors(tfm, X_val_scaled, config, desc="Val")
    
    scorer = AnomalyScorer()
    # Move scorer to device for calculation (though TimesFM output is numpy, we use torch for scorer)
    scorer_device = torch.device("cuda" if config['device'] == "gpu" else "cpu")
    
    if len(val_errors) > 0:
        val_errors_tensor = torch.from_numpy(val_errors).float().to(scorer_device)
        scorer.find_distribution(val_errors_tensor)
    else:
        logger.error("No validation errors computed.")
        return

    # 2. Run Inference on Test Data
    logger.info(f"Running inference on Test set ({len(X_test_scaled)} samples)...")
    test_errors, test_forecasts = run_inference_for_errors(tfm, X_test_scaled, config, desc="Test")
    
    if len(test_errors) == 0:
        logger.error("No test errors computed.")
        return
        
    # 3. Calculate Scores
    test_errors_tensor = torch.from_numpy(test_errors).float().to(scorer_device)
    scores_tensor = scorer.forward(test_errors_tensor)
    
    final_scores = scores_tensor.mean(dim=1).cpu().numpy()
    
    # Align scores (Pad beginning)
    pad_width = config['context_length']
    padding = np.zeros(pad_width)
    full_scores = np.concatenate([padding, final_scores])
    
    # Truncate or Pad
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
            
        # Process forecasts
        test_forecasts_original = scaler.inverse_transform(test_forecasts)
        
        pad_width = config['context_length']
        padding_forecast = np.full((pad_width, test_forecasts_original.shape[1]), np.nan)
        full_forecasts = np.concatenate([padding_forecast, test_forecasts_original])
        
        if len(full_forecasts) < len(df_test):
            extra_pad = np.full((len(df_test) - len(full_forecasts), full_forecasts.shape[1]), np.nan)
            full_forecasts = np.concatenate([full_forecasts, extra_pad])
        elif len(full_forecasts) > len(df_test):
            full_forecasts = full_forecasts[:len(df_test)]

        result_df = pd.DataFrame({
            'timestamp': df_test['timestamp'],
            'score': full_scores,
            'label': y_true
        })
        
        for idx, col_name in enumerate(feature_cols):
            result_df[f'pred_{col_name}'] = full_forecasts[:, idx]

        result_df.to_csv("timesfm_ad_results.csv", index=False)
        logger.info("Results saved to timesfm_ad_results.csv")

if __name__ == "__main__":
    main()
