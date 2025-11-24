"""
Utility functions for demand imputation
"""
import torch
import numpy as np
import sys

sys.path.append('/home/guest/DemandForecasting')
from latent_demand_recovery.model.dlinear import Model

class Config:
    def __init__(self):
        self.seq_len = 480
        self.pred_len = 480
        self.enc_in = 6
        self.individual = True  # Model was trained with individual=True


def load_imputation_model(checkpoint_path='latent_demand_recovery/checkpoints/best_imputation_model.pth', 
                          device='cuda'):
    """
    Load the trained imputation model
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: 'cuda' or 'cpu'
        
    Returns:
        model: Loaded model in eval mode
        device: torch device
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    config = Config()
    model = Model(config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model, device


def impute_sequences(model, X, device, batch_size=256):
    """
    Impute missing values in sequences
    
    Args:
        model: Trained imputation model
        X: numpy array [N, seq_len, features] with NaN for missing values
        device: torch device
        batch_size: batch size for processing
        
    Returns:
        imputed: numpy array with missing values filled
        mask: boolean mask of imputed positions
    """
    model.eval()
    n_samples = X.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    imputed_data = np.zeros_like(X)
    imputation_mask = np.zeros((X.shape[0], X.shape[1]), dtype=bool)
    
    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            # Get batch
            batch = X[start_idx:end_idx]
            X_tensor = torch.FloatTensor(batch).to(device)
            
            # Create mask for missing values
            missing_mask = torch.isnan(X_tensor[:, :, 0])
            
            # Replace NaN with 0 for model input
            X_input = torch.nan_to_num(X_tensor, nan=0.0)
            
            # Get imputation
            output = model.imputation(X_input)
            
            # Fill in missing values
            X_imputed = X_tensor.clone()
            X_imputed[:, :, 0] = torch.where(
                missing_mask,
                output[:, :, 0],
                X_tensor[:, :, 0]
            )
            
            imputed_data[start_idx:end_idx] = X_imputed.cpu().numpy()
            imputation_mask[start_idx:end_idx] = missing_mask.cpu().numpy()
    
    return imputed_data, imputation_mask


def quick_impute(X, checkpoint_path='latent_demand_recovery/checkpoints/best_imputation_model.pth'):
    """
    Quick one-line imputation function
    
    Args:
        X: numpy array with NaN values to impute
        checkpoint_path: path to model checkpoint
        
    Returns:
        imputed: numpy array with missing values filled
    """
    model, device = load_imputation_model(checkpoint_path)
    imputed, _ = impute_sequences(model, X, device)
    return imputed


def compute_imputation_metrics(imputed, ground_truth, mask):
    """
    Compute metrics for imputed values vs ground truth
    
    Args:
        imputed: Imputed data
        ground_truth: Ground truth data
        mask: Boolean mask of imputed positions
        
    Returns:
        metrics: Dictionary of metrics
    """
    imputed_values = imputed[:, :, 0][mask]
    true_values = ground_truth[:, :, 0][mask]
    
    mae = np.abs(imputed_values - true_values).mean()
    rmse = np.sqrt(np.mean((imputed_values - true_values) ** 2))
    mape = np.mean(np.abs((true_values - imputed_values) / (true_values + 1e-8))) * 100
    
    # Correlation
    correlation = np.corrcoef(imputed_values, true_values)[0, 1]
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'correlation': correlation,
        'n_imputed': int(mask.sum())
    }
