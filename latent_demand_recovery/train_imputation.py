import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import sys
import json
import os
from datetime import datetime
sys.path.append('/home/guest/DemandForecasting')
from data_utils.load_data import load_data
from latent_demand_recovery.model.dlinear import Model
from tqdm import tqdm

class ImputationDataset(Dataset):
    def __init__(self, X, ts_origin, valid_idx):
        self.X = torch.FloatTensor(X)
        self.ts_origin = torch.FloatTensor(ts_origin)
        self.valid_idx = torch.BoolTensor(valid_idx)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.ts_origin[idx], self.valid_idx[idx]

class Config:
    def __init__(self):
        self.seq_len = 480
        self.pred_len = 480
        self.enc_in = 6
        self.individual = True

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X, target, valid_idx in dataloader:
        X, target = X.to(device), target.to(device)
        
        # Handle NaN: replace with 0 for model input
        X_input = torch.nan_to_num(X, nan=0.0)
        
        # Get imputation
        output = model.imputation(X_input)
        
        # Loss only on missing positions
        missing_mask = torch.isnan(X[:, :, 0:1])
        loss = criterion(output[:, :, 0:1][missing_mask], target[missing_mask])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    mae_total = 0
    
    with torch.no_grad():
        for X, target, valid_idx in dataloader:
            X, target, valid_idx = X.to(device), target.to(device), valid_idx.to(device)
            
            X_input = torch.nan_to_num(X, nan=0.0)
            output = model.imputation(X_input)
            
            # Evaluate on artificially masked positions
            loss = criterion(output[:, :, 0:1][valid_idx], target[valid_idx])
            mae = torch.abs(output[:, :, 0:1][valid_idx] - target[valid_idx]).mean()
            
            total_loss += loss.item()
            mae_total += mae.item()
    
    return total_loss / len(dataloader), mae_total / len(dataloader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    data = load_data()
    X = data['train_set']['X']
    ts_origin = data['ts_origin']
    valid_idx = data['valid_idx']
    
    # Split train/val/test
    dataset = ImputationDataset(X, ts_origin, valid_idx)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Initialize model
    config = Config()
    model = Model(config).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = nn.MSELoss()
    
    # Training loop
    num_epochs = 50
    best_val_loss = float('inf')
    best_epoch = 0
    
    train_loss_history = []
    val_loss_history = []
    val_mae_history = []
    
    patience = 10
    patience_counter = 0
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae = evaluate(model, val_loader, criterion, device)
        
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        val_mae_history.append(val_mae)
        
        print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f}')
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'latent_demand_recovery/checkpoints/best_imputation_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Test evaluation
    model.load_state_dict(torch.load('latent_demand_recovery/checkpoints/best_imputation_model.pth'))
    test_loss, test_mae = evaluate(model, test_loader, criterion, device)
    print(f'\nTest Loss: {test_loss:.4f} | Test MAE: {test_mae:.4f}')
    
    # Save logs
    os.makedirs('latent_demand_recovery/logs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save training history
    training_history = {
        'train_loss': train_loss_history,
        'val_loss': val_loss_history,
        'val_mae': val_mae_history
    }
    with open(f'latent_demand_recovery/logs/training_history_{timestamp}.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Save overall summary
    summary = {
        'timestamp': timestamp,
        'dataset': {
            'total_samples': len(dataset),
            'train_samples': train_size,
            'val_samples': val_size,
            'test_samples': test_size,
            'sequence_length': config.seq_len,
            'num_features': config.enc_in
        },
        'model_config': {
            'model_type': 'DLinear',
            'seq_len': config.seq_len,
            'pred_len': config.pred_len,
            'enc_in': config.enc_in,
            'individual': config.individual
        },
        'training_config': {
            'num_epochs': num_epochs,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'optimizer': 'Adam',
            'criterion': 'MSELoss',
            'weight_decay': 1e-4,
            'lr_scheduler': 'ReduceLROnPlateau',
            'early_stopping_patience': 10
        },
        'results': {
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'final_train_loss': train_loss_history[-1],
            'final_val_loss': val_loss_history[-1],
            'final_val_mae': val_mae_history[-1],
            'test_loss': test_loss,
            'test_mae': test_mae
        }
    }
    with open(f'latent_demand_recovery/logs/summary_{timestamp}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f'\nLogs saved to latent_demand_recovery/logs/')
    print(f'  - training_history_{timestamp}.json')
    print(f'  - summary_{timestamp}.json')

if __name__ == '__main__':
    main()
