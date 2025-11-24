import torch
import numpy as np
import sys
import json
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import os

sys.path.append('/home/guest/DemandForecasting')
from data_utils.load_data import load_data
from latent_demand_recovery.model.dlinear import Model

class Config:
    def __init__(self):
        self.seq_len = 480
        self.pred_len = 480
        self.enc_in = 6
        self.individual = True  # Model was trained with individual=True

class Imputer:
    def __init__(self, checkpoint_path, device='cuda'):
        """
        Initialize the imputation model
        
        Args:
            checkpoint_path: Path to the trained model checkpoint
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.config = Config()
        self.model = Model(self.config).to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()
        
        print(f"Model loaded from {checkpoint_path}")
        print(f"Using device: {self.device}")
    
    def impute_batch(self, X_batch):
        """
        Impute missing values in a batch of sequences
        
        Args:
            X_batch: numpy array of shape [batch, seq_len, features] with NaN for missing values
            
        Returns:
            imputed: numpy array with missing values filled in
            mask: boolean array indicating which values were imputed
        """
        with torch.no_grad():
            # Convert to tensor
            X_tensor = torch.FloatTensor(X_batch).to(self.device)
            
            # Create mask for missing values (only for the first feature - sales)
            missing_mask = torch.isnan(X_tensor[:, :, 0])
            
            # Replace NaN with 0 for model input
            X_input = torch.nan_to_num(X_tensor, nan=0.0)
            
            # Get imputation
            output = self.model.imputation(X_input)
            
            # Fill in missing values only for the sales channel (first feature)
            X_imputed = X_tensor.clone()
            X_imputed[:, :, 0] = torch.where(
                missing_mask,
                output[:, :, 0],  # Use model prediction
                X_tensor[:, :, 0]  # Keep original value
            )
            
            return X_imputed.cpu().numpy(), missing_mask.cpu().numpy()
    
    def impute_dataset(self, X, batch_size=256):
        """
        Impute an entire dataset
        
        Args:
            X: numpy array of shape [N, seq_len, features]
            batch_size: batch size for processing
            
        Returns:
            imputed_data: numpy array with missing values filled
            imputation_mask: boolean mask showing which values were imputed
        """
        n_samples = X.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        imputed_data = np.zeros_like(X)
        imputation_mask = np.zeros((X.shape[0], X.shape[1]), dtype=bool)
        
        print(f"Imputing {n_samples} sequences in {n_batches} batches...")
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            batch = X[start_idx:end_idx]
            imputed_batch, mask_batch = self.impute_batch(batch)
            
            imputed_data[start_idx:end_idx] = imputed_batch
            imputation_mask[start_idx:end_idx] = mask_batch
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {end_idx}/{n_samples} sequences")
        
        print("Imputation complete!")
        return imputed_data, imputation_mask
    
    def visualize_imputation(self, original, imputed, mask, sample_idx=0, save_path=None):
        """
        Visualize the imputation results for a sample
        
        Args:
            original: Original data with NaN
            imputed: Imputed data
            mask: Boolean mask of imputed positions
            sample_idx: Which sample to visualize
            save_path: Optional path to save the figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(15, 8))
        
        # Extract the sales channel (first feature)
        original_sales = original[sample_idx, :, 0]
        imputed_sales = imputed[sample_idx, :, 0]
        sample_mask = mask[sample_idx]
        
        # Plot 1: Original data with missing values highlighted
        ax1 = axes[0]
        observed_indices = ~np.isnan(original_sales)
        missing_indices = np.isnan(original_sales)
        
        ax1.plot(np.where(observed_indices)[0], original_sales[observed_indices], 
                 'b-', label='Observed', linewidth=1.5, alpha=0.7)
        ax1.scatter(np.where(missing_indices)[0], 
                   np.full(missing_indices.sum(), np.nanmean(original_sales)),
                   c='red', s=20, label='Missing', alpha=0.5, marker='x')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Sales')
        ax1.set_title('Original Data with Missing Values')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Imputed data
        ax2 = axes[1]
        ax2.plot(imputed_sales, 'g-', label='Imputed', linewidth=1.5, alpha=0.7)
        ax2.scatter(np.where(sample_mask)[0], imputed_sales[sample_mask],
                   c='orange', s=30, label='Imputed values', alpha=0.8, marker='o')
        ax2.plot(np.where(observed_indices)[0], original_sales[observed_indices],
                'b.', label='Original observed', markersize=3, alpha=0.5)
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Sales')
        ax2.set_title('Imputed Complete Time Series')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
        
    def compute_statistics(self, original, imputed, mask, ground_truth=None):
        """
        Compute statistics about the imputation
        
        Args:
            original: Original data with NaN
            imputed: Imputed data
            mask: Boolean mask of imputed positions
            ground_truth: Optional ground truth for artificially masked data
            
        Returns:
            stats: Dictionary of statistics
        """
        stats = {}
        
        # Count missing values
        total_values = mask.size
        missing_values = mask.sum()
        stats['total_values'] = int(total_values)
        stats['missing_values'] = int(missing_values)
        stats['missing_percentage'] = float(missing_values / total_values * 100)
        
        # Statistics on imputed values
        imputed_values = imputed[:, :, 0][mask]
        stats['imputed_mean'] = float(np.mean(imputed_values))
        stats['imputed_std'] = float(np.std(imputed_values))
        stats['imputed_min'] = float(np.min(imputed_values))
        stats['imputed_max'] = float(np.max(imputed_values))
        
        # Statistics on observed values
        observed_mask = ~np.isnan(original[:, :, 0])
        observed_values = original[:, :, 0][observed_mask]
        stats['observed_mean'] = float(np.mean(observed_values))
        stats['observed_std'] = float(np.std(observed_values))
        stats['observed_min'] = float(np.min(observed_values))
        stats['observed_max'] = float(np.max(observed_values))
        
        # If ground truth is available, compute accuracy metrics
        if ground_truth is not None:
            gt_values = ground_truth[:, :, 0][mask]
            mae = np.abs(imputed_values - gt_values).mean()
            rmse = np.sqrt(np.mean((imputed_values - gt_values) ** 2))
            mape = np.mean(np.abs((gt_values - imputed_values) / (gt_values + 1e-8))) * 100
            
            stats['mae'] = float(mae)
            stats['rmse'] = float(rmse)
            stats['mape'] = float(mape)
        
        return stats


def main():
    """
    Main function to run the imputation process
    """
    print("=" * 60)
    print("Demand Imputation Process")
    print("=" * 60)
    
    # Initialize imputer
    checkpoint_path = 'latent_demand_recovery/checkpoints/best_imputation_model.pth'
    imputer = Imputer(checkpoint_path, device='cuda')
    
    # Load data
    print("\nLoading data...")
    data = load_data()
    X = data['train_set']['X']
    ts_origin = data['ts_origin']
    valid_idx = data['valid_idx']
    
    print(f"Data shape: {X.shape}")
    print(f"  - Samples: {X.shape[0]}")
    print(f"  - Sequence length: {X.shape[1]}")
    print(f"  - Features: {X.shape[2]}")
    
    # For demo, let's use a subset
    subset_size = 1000
    X_subset = X[:subset_size]
    ts_origin_subset = ts_origin[:subset_size]
    valid_idx_subset = valid_idx[:subset_size]
    
    print(f"\nUsing subset of {subset_size} samples for demonstration")
    
    # Perform imputation
    print("\n" + "=" * 60)
    print("Starting imputation...")
    print("=" * 60)
    imputed_data, imputation_mask = imputer.impute_dataset(X_subset, batch_size=256)
    
    # Compute statistics
    print("\n" + "=" * 60)
    print("Imputation Statistics")
    print("=" * 60)
    stats = imputer.compute_statistics(
        X_subset, 
        imputed_data, 
        imputation_mask,
        ground_truth=ts_origin_subset  # We have ground truth for evaluation
    )
    
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key:25s}: {value:.6f}")
        else:
            print(f"{key:25s}: {value}")
    
    # Save results
    print("\n" + "=" * 60)
    print("Saving Results")
    print("=" * 60)
    
    os.makedirs('latent_demand_recovery/imputation_results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save statistics
    stats_path = f'latent_demand_recovery/imputation_results/stats_{timestamp}.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Statistics saved to {stats_path}")
    
    # Save imputed data (subset for demo)
    save_samples = 100
    np.savez_compressed(
        f'latent_demand_recovery/imputation_results/imputed_data_{timestamp}.npz',
        original=X_subset[:save_samples],
        imputed=imputed_data[:save_samples],
        mask=imputation_mask[:save_samples],
        ground_truth=ts_origin_subset[:save_samples]
    )
    print(f"✓ Imputed data (first {save_samples} samples) saved")
    
    # Visualize a few examples
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    
    os.makedirs('latent_demand_recovery/imputation_results/figures', exist_ok=True)
    
    # Visualize 3 random samples
    np.random.seed(42)
    sample_indices = np.random.choice(subset_size, size=min(3, subset_size), replace=False)
    
    for i, idx in enumerate(sample_indices):
        fig_path = f'latent_demand_recovery/imputation_results/figures/sample_{i+1}_{timestamp}.png'
        print(f"\nVisualizing sample {i+1} (index {idx})...")
        imputer.visualize_imputation(
            X_subset, 
            imputed_data, 
            imputation_mask, 
            sample_idx=idx,
            save_path=fig_path
        )
    
    print("\n" + "=" * 60)
    print("Imputation Complete!")
    print("=" * 60)
    print(f"\nResults saved to: latent_demand_recovery/imputation_results/")
    print(f"  - Statistics: stats_{timestamp}.json")
    print(f"  - Imputed data: imputed_data_{timestamp}.npz")
    print(f"  - Figures: figures/sample_*_{timestamp}.png")
    
    return imputed_data, imputation_mask, stats


if __name__ == '__main__':
    imputed_data, imputation_mask, stats = main()
