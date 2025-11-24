import torch
from torch.utils.data import Dataset
from pygrinder import mcar, fill_and_get_mask_torch

class BaseDataset(Dataset):
    def __init__(self, data, OT=1, rate=0.2):
        self.X = data['X']
        self.OT = OT
        self.rate = rate

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X_ori = self.X[idx] # x chứa nan từ load_data
        X = torch.cat([mcar(X_ori[:,:self.OT], p=self.rate), X_ori[:,self.OT:]], dim=-1) 
        X, missing_mask = fill_and_get_mask_torch(X)
        X_ori, X_ori_missing_mask = fill_and_get_mask_torch(X_ori)
        indicating_mask = (X_ori_missing_mask - missing_mask).to(torch.float32)
        
        sample = [
                torch.tensor(idx),
                X,
                missing_mask,
                X_ori,
                indicating_mask,
            ]
        
        return sample
