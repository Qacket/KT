import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
class KT(Dataset):
    def __init__(self, data_dir, crowd_dir):

        crowd_data = np.load(data_dir + crowd_dir, allow_pickle=True)

        self.task_id = crowd_data[:, 0:1]
        self.task_inputs = crowd_data[:, 1:1001]
        self.annotator_id = crowd_data[:, 1001:1002]
        self.annotator_inputs = crowd_data[:, 1002:2002]
        self.label = crowd_data[:, 2002:2003]

    def __len__(self):
        return len(self.task_id)

    def __getitem__(self, idx):
        return idx, self.task_id[idx], self.task_inputs[idx], self.annotator_id[idx], self.annotator_inputs[idx], self.label[idx]