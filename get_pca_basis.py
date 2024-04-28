import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import time
import matplotlib.pyplot as plt
device = torch.device("cpu")
torch.cuda.set_device(2)
data_x = torch.from_numpy(np.load("x_data.npy", allow_pickle = True)).to(device)[:, :, :, :]
# data_x_2 = torch.from_numpy(np.load("x_data_2.npy", allow_pickle = True)).to(device)[:, :, :, :]
# data_x_supplement = torch.from_numpy(np.load("x_data_supplemental.npy", allow_pickle = True)).to(device)[:, :, :, :]
print("finished loading")
# data_x = torch.concatenate((data_x, data_x_2), dim = 0)
data_x = data_x.reshape(data_x.shape[0], -1)
# print(data_x.shape)
data_x_mean = torch.mean(data_x, dim = 0, keepdim = True)
# data_x_std = torch.std(data_x, dim = 0, keepdim = True)
data_x = (data_x - data_x_mean)
U, S, V = torch.pca_lowrank(data_x, q=1000, center=True, niter=10)
plt.plot(S, "k-")
plt.show()
plt.savefig("shallow-water-svd.png")
# np.save("PCA_V.npy", V[:, :100].detach().cpu().numpy(), allow_pickle=True)
# np.save("PCA_means.npy", data_x_mean.detach().cpu().numpy(), allow_pickle=True)