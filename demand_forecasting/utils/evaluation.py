import torch
from tqdm import tqdm
from models._models import arima_forecast
from .metrics import compute_metrics


def evaluate_neural_model(model, dataloader, criterion, device, desc="Eval"):
    """Evaluate neural network model"""
    model.eval()
    total_loss = 0
    all_metrics = {'wape': 0, 'wpe': 0}
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=desc, leave=False)
        for x, y in pbar:
            x = x.to(device).float()
            y = y.to(device).float()
            
            # Forward
            pred = model(x)
            
            # Handle shape
            if pred.dim() == 2 and pred.shape[1] == 1:
                pred = pred.unsqueeze(-1)
            
            if y.dim() == 1:
                y = y.unsqueeze(1).unsqueeze(1)
            elif y.dim() == 2:
                y = y.unsqueeze(1)
            
            # Compute loss
            loss = criterion(pred, y)
            
            # Metrics
            total_loss += loss.item()
            batch_metrics = compute_metrics(pred, y)
            for k, v in batch_metrics.items():
                all_metrics[k] += v
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'wape': f"{batch_metrics['wape']:.2f}%"
            })
    
    # Average
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    for k in all_metrics:
        all_metrics[k] /= num_batches
    
    return avg_loss, all_metrics


def evaluate_arima(dataloader, criterion, device, arima_params, desc="ARIMA"):
    """Evaluate ARIMA model"""
    total_loss = 0
    all_metrics = {'wape': 0, 'wpe': 0}
    
    p, d, q = arima_params
    pbar = tqdm(dataloader, desc=desc, leave=False)
    
    for x, y in pbar:
        x = x.to(device).float()
        y = y.to(device).float()
        
        # ARIMA expects (batch, seq_len)
        x_squeezed = x.squeeze(-1)
        
        # Forecast
        pred = arima_forecast(x_squeezed, p=p, d=d, q=q, steps=1)
        
        # Handle shape
        if pred.dim() == 1:
            pred = pred.unsqueeze(1).unsqueeze(1)
        elif pred.dim() == 2:
            pred = pred.unsqueeze(1)
        
        if y.dim() == 1:
            y = y.unsqueeze(1).unsqueeze(1)
        elif y.dim() == 2:
            y = y.unsqueeze(1)
        
        # Compute loss
        loss = criterion(pred, y)
        
        # Metrics
        total_loss += loss.item()
        batch_metrics = compute_metrics(pred, y)
        for k, v in batch_metrics.items():
            all_metrics[k] += v
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'wape': f"{batch_metrics['wape']:.2f}%"
        })
    
    # Average
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    for k in all_metrics:
        all_metrics[k] /= num_batches
    
    return avg_loss, all_metrics
