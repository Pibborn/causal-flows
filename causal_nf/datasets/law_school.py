import torch 

class LawSchoolDataset:

    def __init__(self, x: torch.Tensor):
        self.x = x
        IDX_SEX = 1
        IDX_RACE = 0
        self.binary_dims = torch.tensor([IDX_SEX, IDX_RACE])
        self.binary_min_values  = torch.zeros(2)
        self.binary_max_values  = torch.ones(2)

    def __len__(self):                 return self.x.shape[0]
    def __getitem__(self, idx):        return self.x[idx]

    def data(self, scaler=None, **kw):
        if scaler is not None and hasattr(scaler, "transform"):
            return scaler.transform(self.x), None
        return self.x, None

    def set_add_noise(self, add_noise: bool): pass

    def prepare_data(self): pass