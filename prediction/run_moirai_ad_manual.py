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

# Check for MOIRAI / uni2ts
try:
    from uni2ts.model.moirai import MoiraiModule
except ImportError:
    logger.error("uni2ts package not found.")
    logger.error("Please install it following instructions at: https://github.com/SalesforceAIResearch/uni2ts")
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
class MoiraiDataset(Dataset):
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
        
        # MOIRAI expects specific dictionary format for batching
        # We construct the inputs required for the forward/predict pass
        # Note: Dimensions usually need to be (Batch, Time, Variates)
        # Here we return (Time,) and will stack in collate_fn or DataLoader
        
        return context, target

def run_inference_for_errors(module, data_array, config, desc="Inference"):
    """
    Run MOIRAI inference on a multivariate data array.
    """
    n_samples, n_features = data_array.shape
    feature_errors = []
    
    device = config['device']
    
    for i in range(n_features):
        data_col = data_array[:, i]
        logger.info(f"Processing feature {i+1}/{n_features}...")
        
        dataset = MoiraiDataset(
            data_col, 
            context_length=config['context_length'], 
            prediction_length=config['prediction_length'],
            stride=config['stride']
        )
        
        if len(dataset) == 0:
            logger.warning(f"Dataset empty for feature {i}.")
            feature_errors.append(np.zeros((0,)))
            continue

        loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
        
        col_errors = []
        
        with torch.no_grad():
            for context, target in tqdm(loader, desc=f"{desc} Feat {i}", leave=False):
                # context: (batch, context_len)
                # target: (batch, pred_len)
                
                batch_size = context.shape[0]
                
                # Prepare inputs for MOIRAI
                # MOIRAI expects:
                # - past_target: (batch, context_len, variates)
                # - past_observed_mask: (batch, context_len, variates)
                # - past_is_pad: (batch, context_len)
                
                # Add variate dimension (univariate -> 1 variate)
                past_target = context.unsqueeze(-1).to(device) # (B, T, 1)
                past_observed_mask = torch.ones_like(past_target).to(device) # (B, T, 1)
                past_is_pad = torch.zeros(batch_size, config['context_length']).to(device) # (B, T)
                
                # MOIRAI Inference
                # Note: The exact API depends on uni2ts version. 
                # We assume a standard forward/predict interface.
                # Often MOIRAI returns a distribution object.
                
                # We need to specify prediction_length and num_samples if using a generate method
                # Or if using forward, it might return loss.
                # Let's try to use the module's forecast capability if exposed, 
                # otherwise we might need to construct the input for 'predict_step'
                
                # Construct batch dict
                batch = {
                    "past_target": past_target,
                    "past_observed_mask": past_observed_mask,
                    "past_is_pad": past_is_pad,
                    "prediction_length": config['prediction_length']
                }
                
                # Some versions of MOIRAI Module might need 'num_samples' in the call
                # dist = module(batch, num_samples=20) 
                # However, MoiraiModule usually returns a distribution on the *next* steps given the past.
                
                # If the module is a LightningModule, it might not be callable directly for inference without 'forward' logic.
                # We assume module(batch) returns the forecast distribution for the prediction horizon.
                
                dist = module(batch)
                
                # Sample or get mean
                # dist shape usually: (batch, samples, prediction_len, variates) or (batch, prediction_len, variates)
                # If it's a distribution object (like StudentT), we can take .mean or .sample()
                
                if hasattr(dist, 'mean'):
                    forecast = dist.mean # (batch, pred_len, variates)
                elif hasattr(dist, 'sample'):
                    samples = dist.sample((20,)) # (samples, batch, pred_len, variates)
                    forecast = samples.median(dim=0).values
                else:
                    # Fallback if it returns a tensor directly
                    forecast = dist
                
                # Squeeze variate dim if present
                if forecast.dim() == 3:
                    forecast = forecast.squeeze(-1) # (batch, pred_len)
                
                # Calculate Error (MAE)
                target = target.to(device)
                error = torch.abs(forecast - target)
                error = error.mean(dim=1) # Average over horizon
                
                col_errors.append(error.cpu())
        
        if col_errors:
            feature_errors.append(torch.cat(col_errors).numpy())
        else:
            feature_errors.append(np.array([]))
            
    if not feature_errors or len(feature_errors[0]) == 0:
        return np.array([])
        
    return np.stack(feature_errors, axis=1)

# ==========================================
# 3. Main
# ==========================================
def main():
    # Configuration
    config = {
        'model_name': "Salesforce/moirai-1.0-R-small", 
        'context_length': 512,        
        'prediction_length': 1,       
        'batch_size': 32,             
        'stride': 1,                  
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'data_dir': "data/preprocessed/multivariate/ESA-Mission1-semi-supervised",
        'train_file': "3_months.train.csv",
        'test_file': "84_months.test.csv",
        'split_date': "2000-03-11",
        'patch_size': 32 # MOIRAI specific
    }
    
    train_path = os.path.join(config['data_dir'], config['train_file'])
    test_path = os.path.join(config['data_dir'], config['test_file'])
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        logger.error("Data files not found.")
        return

    # Load Model
    logger.info(f"Loading model: {config['model_name']} on {config['device']}")
    
    # Load MOIRAI Module
    module = MoiraiModule.from_pretrained(config['model_name'])
    module.to(config['device'])
    module.eval()

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
    val_errors = run_inference_for_errors(module, X_val_scaled, config, desc="Val")
    
    scorer = AnomalyScorer()
    if len(val_errors) > 0:
        val_errors_tensor = torch.from_numpy(val_errors).float().to(config['device'])
        scorer.find_distribution(val_errors_tensor)
    else:
        logger.error("No validation errors computed.")
        return

    # 2. Run Inference on Test Data
    logger.info(f"Running inference on Test set ({len(X_test_scaled)} samples)...")
    test_errors = run_inference_for_errors(module, X_test_scaled, config, desc="Test")
    
    if len(test_errors) == 0:
        logger.error("No test errors computed.")
        return
        
    # 3. Calculate Scores
    test_errors_tensor = torch.from_numpy(test_errors).float().to(config['device'])
    scores_tensor = scorer.forward(test_errors_tensor)
    
    final_scores = scores_tensor.mean(dim=1).cpu().numpy()
    
    # Align scores
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
            
        result_df = pd.DataFrame({
            'timestamp': df_test['timestamp'],
            'score': full_scores,
            'label': y_true
        })
        result_df.to_csv("moirai_ad_results.csv", index=False)
        logger.info("Results saved to moirai_ad_results.csv")

if __name__ == "__main__":
    main()
