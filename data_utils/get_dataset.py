from torch.utils.data import Dataset
import torch
from pygrinder import fill_and_get_mask_torch, mcar

import os
os.environ['SCIPY_ARRAY_API'] = '1'

class CustomDataset(Dataset):
    def __init__(self, data, return_X_ori=False, rate=0.2, OT=1):
        super().__init__()
        self.data = data
        self.return_X_ori = return_X_ori
        self.rate = rate
        self.OT = OT
        
        X = data["X"] if isinstance(data['X'], torch.Tensor) else torch.tensor(data["X"])
        X_ori = None if "X_ori" not in data.keys() else data["X_ori"] if isinstance(data['X_ori'], torch.Tensor) else torch.tensor(data["X_ori"])
        
        if return_X_ori:
            self.X, self.missing_mask = fill_and_get_mask_torch(X)
            self.X_ori, X_ori_missing_mask = fill_and_get_mask_torch(X_ori)
            indicating_mask = X_ori_missing_mask - self.missing_mask
            self.indicating_mask = indicating_mask.to(torch.float32)
        else:
            self.X, self.mask = fill_and_get_mask_torch(X)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        if self.return_X_ori:
            X = self.X[idx]
            X_ori = self.X_ori[idx]
            missing_mask = self.missing_mask[idx]
            indicating_mask = self.indicating_mask[idx]
        else:
            X_ori = self.X[idx]
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
        