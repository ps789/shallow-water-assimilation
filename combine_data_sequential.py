import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from tqdm import tqdm
data_x = []
data_y = []
data_t = []
for i in tqdm(range(40)):
    data_x.append(np.load(f"./sequential_data/x_data_sequential_{i}.npy", allow_pickle = True))
    data_y.append(np.load(f"./sequential_data/y_data_sequential_{i}.npy", allow_pickle = True))
    data_t.append(np.load(f"./sequential_data/t_data_sequential_{i}.npy", allow_pickle = True))
data_x = np.concatenate(data_x, axis = 0)
data_y = np.concatenate(data_y, axis = 0)
data_t = np.concatenate(data_t, axis = 0)
print("done concat")
np.save(f"./sequential_data/x_data_sequential.npy", data_x, allow_pickle=True)
np.save(f"./sequential_data/y_data_sequential.npy", data_y, allow_pickle=True)
np.save(f"./sequential_data/t_data_sequential.npy", data_t, allow_pickle=True)
