from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from torch.distributions import Independent, Normal

from causal_nf.preparators.tabular_preparator import TabularPreparator
from causal_nf.datasets.law_school import LawSchoolDataset
from causal_nf.sem_equations import sem_dict
from causal_nf.utils.io import dict_to_cn
from causal_nf.utils.scalers import StandardTransform
from causal_nf.config import ROOT

IDX_SEX, IDX_RACE, IDX_UGPA, IDX_LSAT = 0, 1, 2, 3


class LawSchoolPreparator(TabularPreparator):

    def __init__(self, add_noise, **kwargs):
        self.add_noise = add_noise
        self.dataset = None

        sem_fn = sem_dict["law_school"](sem_name="dummy") # identity SEM
        self.adjacency = sem_fn.adjacency
        self.num_nodes = 4
        self.intervention_index_list = [IDX_SEX, IDX_RACE]
        self.sensitive_idx = IDX_SEX

        super().__init__(name="law_school", task="modeling", **kwargs)
        if len(self.split) < 3: # last-minute fix? don't know why this is not handled correctly :) 
            self.split = [0.8, 0.1, 0.1]
        #assert self.split == [0.8, 0.1, 0.1]

    @classmethod
    def params(cls, dataset):
        if isinstance(dataset, dict):
            dataset = dict_to_cn(dataset)
        d = {"add_noise": dataset.add_noise}
        d.update(TabularPreparator.params(dataset))
        return d

    @classmethod
    def loader(cls, dataset):
        return cls(**cls.params(dataset))

    def _x_dim(self):
        return self.num_nodes                       # 4

    def _loss(self, loss):
        if loss in ["default", "forward"]:
            return "forward"
        raise NotImplementedError

    def _get_dataset_raw(self):
        csv = Path(ROOT, 'data', 'clean_LawSchool.csv')
        df  = pd.read_csv(csv, sep="|")
        df = df.rename(columns=str.lower)[["sex", "race", "ugpa", "lsat"]]
        df["sex"]  = (df["sex"] == "Male").astype(float)
        df["race"] = (df["race"] != "White").astype(float)
        return df.astype(float)

    
    def _split_dataset(self, df: pd.DataFrame):
        """
        Splits the DataFrame into train/val/test according to self.split
        and wraps each slice in LawSchoolDataset.
        """
        n        = len(df)
        n_train  = int(self.split[0] * n)
        n_val    = int(self.split[1] * n)

        def to_ds(frame):
            return LawSchoolDataset(torch.tensor(frame.values, dtype=torch.float32))

        ds_train = to_ds(df.iloc[:n_train])
        ds_val   = to_ds(df.iloc[n_train:n_train + n_val])
        ds_test  = to_ds(df.iloc[n_train + n_val:])

        # keep handle for post_process()
        self.dataset = ds_train
        return [ds_train, ds_val, ds_test]

    @property
    def dims_scaler(self):
        return (0,)

    def get_scaler(self, fit=True):
        scaler = self._get_scaler()
        self.scaler_transform = None
        if fit:
            x = self.get_features_train()
            scaler.fit(x, dims=self.dims_scaler)
            if self.scale in ["default", "std"]:
                self.scaler_transform = StandardTransform(
                    shift=x.mean(0), scale=x.std(0)
                )
        self.scaler = scaler
        return scaler

    def get_scaler_info(self):
        if self.scale in ["default", "std"]:
            return [("std", None)]
        raise NotImplementedError

    def get_intervention_list(self):
        return [{"name": "flip", "value": 1, "index": idx}
                for idx in self.intervention_index_list]

    def intervene(self, index, value, shape):
        if len(shape) == 1:
            shape = (shape[0], self._x_dim())
        x = self.get_features_train()[:shape[0]].clone()
        if index in (IDX_SEX, IDX_RACE):
            x[..., index] = 1 - x[..., index]
        else:
            x[..., index] = value
        return x

    # placeholder counterfactual (will be replaced by trained flow)
    def compute_counterfactual(self, x_factual, index, value):
        x_cf = x_factual.clone()
        if index in (IDX_SEX, IDX_RACE):
            x_cf[:, index] = 1 - x_cf[:, index]
        else:
            x_cf[:, index] = value
        return x_cf

    def log_prob(self, x):
        d = self._x_dim()
        return Independent(Normal(torch.zeros(d), torch.ones(d)), 1).log_prob(x)

    def post_process(self, x, inplace=False):
        if not inplace:
            x = x.clone()
        dims = self.dataset.binary_dims
        x[..., dims] = x[..., dims].floor().float()
        x[..., dims] = torch.clamp(x[..., dims],
                                   min=self.dataset.binary_min_values,
                                   max=self.dataset.binary_max_values)
        return x

    def feature_names(self, latex=False):
        return ["sex", "race", "UGPA", "LSAT"]
