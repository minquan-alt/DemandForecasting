import numpy as np
import pandas as pd
import torch
import argparse
from torch.utils.data import DataLoader

from tqdm import tqdm
import os
import random
import sys
sys.path.append('/home/guest/DemandForecasting')

from model.dlinear import Model
from data_utils.load_data import load_and_preprocess_data

# set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_type', type=str, default='imputed', choices=['imputed', 'original'], help='Data type: imputed or original')
parser.add_argument('--use_decoder', action='store_true', help='Use decoder with future covariates (default: False)')
parser.add_argument('--no_decoder', dest='use_decoder', action='store_false', help='Do not use decoder')
parser.set_defaults(use_decoder=True)
args = parser.parse_args()

# configuration
class Config:
    def __init__(self, data_type='imputed', use_decoder=True):
        # paths
        self.data_type = data_type
        self.data_path = f'/home/guest/DemandForecasting/data/{data_type}_data.csv'
        self.save_path = f'/home/guest/DemandForecasting/demand_forecasting/checkpoints/{data_type}_{"decoder" if use_decoder else "no_decoder"}/'
        self.log_path = f'/home/guest/DemandForecasting/demand_forecasting/logs/{data_type}_{"decoder" if use_decoder else "no_decoder"}/'
        
        # model parameters
        self.model = 'dlinear'
        self.patience = 5
        self.enable_scheduler = True
        self.seq_len = 480
        self.pred_len = 112
        self.enc_in = 10
        self.dec_in = 9
        self.use_decoder = use_decoder
        self.individual = True
        
        # training parameters
        self.batch_size = 1024
        self.lr = 0.001
        self.epochs = 20
        self.train_ratio = 0.99
        
configs = Config(data_type=args.data_type, use_decoder=args.use_decoder)

# load data
dataset, scaler = load_and_preprocess_data(
    data_path=configs.data_path,
    time_encoded=True,
    input_len=configs.seq_len,
    target_len=configs.pred_len
)

print(f'Total dataset size: {len(dataset)}')

# split train/val/test
num_train = int(len(dataset) * configs.train_ratio)
num_val = int(len(dataset) * (1 - configs.train_ratio) / 2)

train_dataset = torch.utils.data.Subset(dataset, list(range(0, num_train)))
val_dataset = torch.utils.data.Subset(dataset, list(range(num_train, num_train + num_val)))
test_dataset = torch.utils.data.Subset(dataset, list(range(num_train + num_val, len(dataset))))

print(f'Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}')
# prepare data loader

train_loader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=False)

# initialize model
model = Model(configs)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# create save directories
os.makedirs(configs.save_path, exist_ok=True)
os.makedirs(configs.log_path, exist_ok=True)

# training loop
best_val_loss = float('inf')
train_loss_history = []
val_loss_history = []
patience_counter = 0

for epoch in tqdm(range(configs.epochs), desc='Epochs'):
    model.train()
    epoch_train_losses = []
    for x, x_dec, y in tqdm(train_loader, desc='Training', leave=False):
        x = x.to(device)
        x_dec = x_dec.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        output = model(x, x_dec)
        if not configs.use_decoder:
            output = output[:, :, -1:]
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        epoch_train_losses.append(loss.item())
    
    avg_train_loss = np.mean(epoch_train_losses)
    train_loss_history.append(avg_train_loss)
    
    model.eval()
    epoch_val_losses = []
    with torch.no_grad():
        for x, x_dec, y in val_loader:
            x = x.to(device)
            x_dec = x_dec.to(device)
            y = y.to(device)
            
            output = model(x, x_dec)
            if not configs.use_decoder:
                output = output[:, :, -1:]
            loss = criterion(output, y)
            epoch_val_losses.append(loss.item())
    
    avg_val_loss = np.mean(epoch_val_losses)
    val_loss_history.append(avg_val_loss)
    
    if configs.enable_scheduler:
        scheduler.step(avg_val_loss)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(configs.save_path, 'best_dlinear_model.pth'))
        print(f"Epoch {epoch+1}/{configs.epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} ✓ (saved)")
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"Epoch {epoch+1}/{configs.epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        if patience_counter >= configs.patience:
            print("Early stopping triggered.")
            break

# save model
train_logs = {
    'train_loss': train_loss_history,
    'val_loss': val_loss_history
}

log_df = pd.DataFrame(train_logs)
log_df.to_csv(os.path.join(configs.log_path, 'dlinear_training_log.csv'), index=False)