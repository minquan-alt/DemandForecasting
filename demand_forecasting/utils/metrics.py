import torch


def compute_metrics(pred, target):
    """Tính các metrics đánh giá: WAPE và WPE"""
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    
    # WAPE: Weighted Absolute Percentage Error
    # WAPE = sum(|actual - forecast|) / sum(|actual|) * 100
    wape = (torch.sum(torch.abs(target - pred)) / torch.sum(torch.abs(target))) * 100
    
    # WPE: Weighted Percentage Error
    # WPE = sum(actual - forecast) / sum(actual) * 100
    wpe = (torch.sum(target - pred) / torch.sum(target)) * 100
    
    return {
        'wape': wape.item(),
        'wpe': wpe.item()
    }
