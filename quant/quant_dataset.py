import torch
from torch.utils.data.dataset import Dataset

class DiffusionInputDataset(Dataset):
    
    def __init__(self, data_path):
        data_list = torch.load(data_path, map_location='cpu') ## its a list of tuples of tensors
        self.xt_list = [] # noisy trajectory, shape = (1,16,10) (batch, horizon, action_dim)
        self.t_list = [] # timestep, shape = 0 (scalar)
        self.y_list = [] # global_cond, shape = (1, 274)

        ## datalist[i][0].shape (B,4,32,32), flat B dimension

        for i, (xt, t, y) in enumerate(data_list): # len = 1522

            # xt : [B, T, Da]
            # assert xt.ndim==3, f"xt must be 3-dim, but got {xt.shape}"
            B = xt.shape[0]

            # t : scalar 
            if isinstance(t, torch.Tensor):
                if t.ndim==0: 
                    t = t.view(1).expand(B) # [B]
            else: 
                t = torch.tensor(t).view(1).expand(B)

            # y : [B,C]
            # assert y.ndim==2 and y.shape[0] == B, f"y shape {y.shape} not compatible with B={B}"

            for b in range(B): 
                self.xt_list.append(xt[b].clone()) #[T,Da]
                self.t_list.append(t[b].long().clone()) # scalar(long)
                self.y_list.append(y[b].clone())


    def __len__(self):
        return len(self.xt_list)
    
    def __getitem__(self, idx):
        return self.xt_list[idx], self.t_list[idx], self.y_list[idx]


class DiffusionInputDatasetLowdimPushT(Dataset):
    
    def __init__(self, data_path):
        data_list = torch.load(data_path, map_location='cpu') ## its a list of tuples of tensors
        self.xt_list = [] # noisy trajectory, shape = (1,16,10) (batch, horizon, action_dim)
        self.t_list = [] # timestep, shape = 0 (scalar)

        ## datalist[i][0].shape (B,4,32,32), flat B dimension
        
        for i, (xt, t, cond) in enumerate(data_list): # len = 1522

            # xt : [B, T, Da]
            assert xt.ndim==3, f"xt must be 3-dim, but got {xt.shape}"
            B = xt.shape[0]

            # t : scalar 
            if isinstance(t, torch.Tensor):
                if t.ndim==0: 
                    t = t.view(1).expand(B) # [B]
            else: 
                t = torch.tensor(t).view(1).expand(B)

            for b in range(B): 
                self.xt_list.append(xt[b].clone()) #[T,Da]
                self.t_list.append(t[b].long().clone()) # scalar(long)


    def __len__(self):
        return len(self.xt_list)
    
    def __getitem__(self, idx):
        return self.xt_list[idx], self.t_list[idx]