import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import time
data_x = np.load("x_data.npy", allow_pickle = True)
data_y = np.load("y_data.npy", allow_pickle = True)
data_t = np.load("t_data.npy", allow_pickle = True)[:, np.newaxis]
data_t = data_t/np.max(data_t)
X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(data_x, data_y, data_t, test_size = 0.2, shuffle = False)#, random_state = 42)

class Conv_Model(nn.Module):
    def __init__(self, hidden_dim):
        super(Conv_Model, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = hidden_dim, kernel_size = 3, padding = 'same'),
            nn.ReLU()
        )
        self.res_blocks = []
        self.t_blocks = []
        for _ in range(5):
            self.res_blocks.append(nn.Sequential(
                nn.Conv2d(in_channels = hidden_dim, out_channels = hidden_dim, kernel_size = 3, padding = "same"),
                nn.ReLU()
            ))
        self.res_blocks = torch.nn.ModuleList(self.res_blocks)
        # self.t_blocks.append(nn.Sequential(
        #     nn.Linear(1, 150*150*hidden_dim)
        # ))
        # self.t_blocks.append(nn.Sequential(
        #     nn.Linear(1, 150*150*hidden_dim)
        # ))
        # self.t_blocks.append(nn.Sequential(
        #     nn.Linear(1, 30*30*hidden_dim)
        # ))
        # self.t_blocks.append(nn.Sequential(
        #     nn.Linear(1, 30*30*hidden_dim)
        # ))
        # self.t_blocks.append(nn.Sequential(
        #     nn.Linear(1, 10*10*hidden_dim)
        # ))
        # self.t_blocks.append(nn.Sequential(
        #     nn.Linear(1, 10*10*hidden_dim)
        # ))
        # self.t_blocks.append(nn.Sequential(
        #     nn.Linear(1, 5*5*hidden_dim)
        # ))
        # self.t_blocks = torch.nn.ModuleList(self.t_blocks)
        self.t_trans = nn.Linear(1, hidden_dim)
        self.hidden_dim = hidden_dim
        self.maxpool_2 = nn.MaxPool2d(kernel_size = 2)
        self.maxpool_3 = nn.MaxPool2d(kernel_size = 3)
        self.maxpool_5 = nn.MaxPool2d(kernel_size = 5)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(in_features = 5*5*hidden_dim, out_features = hidden_dim)
        self.linear_2 = nn.Linear(in_features = hidden_dim, out_features = 2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, t):
        x = self.input_layer(x) + self.relu(self.t_trans(t).view(-1, self.hidden_dim, 1, 1))#self.t_blocks[0](t).view(-1, self.hidden_dim, 150, 150)
        res = x
        x = self.maxpool_5(res + self.res_blocks[0](x))# + self.t_blocks[1](t).view(-1, self.hidden_dim, 150, 150))
        res = x
        x = res + self.res_blocks[1](x)# + self.t_blocks[2](t).view(-1, self.hidden_dim, 30, 30)
        res = x
        x = self.maxpool_3(res + self.res_blocks[2](x))# + self.t_blocks[3](t).view(-1, self.hidden_dim, 30, 30))
        res = x
        x = res + self.res_blocks[2](x)# + self.t_blocks[4](t).view(-1, self.hidden_dim, 10, 10)
        res = x
        x = self.maxpool_2(res + self.res_blocks[3](x))# + self.t_blocks[5](t).view(-1, self.hidden_dim, 10, 10))
        res = x
        x = self.flatten(res + self.res_blocks[4](x))# + self.t_blocks[6](t).view(-1, self.hidden_dim, 5, 5))
        x = self.relu(self.linear_1(x))
        x = self.sigmoid(self.linear_2(x))
        return x
class CustomDataset(Dataset):

    def __init__(self, x_data, y_data, t_data, device):
        self.x_data = torch.Tensor(x_data).to(device)
        self.y_data = torch.Tensor(y_data).to(device)
        self.t_data = torch.Tensor(t_data).to(device)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):

        return self.x_data[idx], self.y_data[idx], self.t_data[idx]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, flush = True)
model = Conv_Model(64)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss = torch.nn.MSELoss()
train_dataset = CustomDataset(X_train, y_train, t_train, device)
valid_dataset = CustomDataset(X_test, y_test, t_test, device)
def train_epoch(model, criterion_cls, optimizer, train_loader):
    losses = []
    model.train()
    with tqdm(total=len(train_loader), desc=f"Train {epoch}: ") as pbar:
        for i, value in enumerate(train_loader):
            x, y, t = value
            target_pred  = model(x, t)
            optimizer.zero_grad()
            batch_loss = criterion_cls(target_pred, y)
            loss = batch_loss

            loss_val = batch_loss.item()
            losses.append(loss_val)
            loss.backward()
            optimizer.step()
            # scheduler.step()
            pbar.update(1)
            pbar.set_postfix_str(
                f"Loss: {loss_val:.3f} ({np.mean(losses):.3f}))")
    return np.mean(losses)
def valid_epoch(model, criterion_cls, valid_loader):
    losses = []
    model.eval()
    with tqdm(total=len(valid_loader), desc=f"Valid {epoch}: ") as pbar:
        for i, value in enumerate(valid_loader):
            x, y, t = value
            target_pred  = model(x, t)
            batch_loss = criterion_cls(target_pred, y)

            loss_val = batch_loss.item()
            losses.append(loss_val)
            # scheduler.step()
            pbar.update(1)
            pbar.set_postfix_str(
                f"Loss: {loss_val:.3f} ({np.mean(losses):.3f}))")
    return np.mean(losses)
def valid_epoch_alt(model, valid_loader):
    model.eval()
    for i, value in enumerate(valid_loader):
        x, y, t = value
        target_pred  = model(x, t)
        print(target_pred)
        print(y, flush = True)
        time.sleep(5)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False)
for epoch in range(200):
    print('EPOCH {}:'.format(epoch + 1))
    print(train_epoch(model, loss, optimizer, train_loader))
    print(valid_epoch(model, loss, valid_loader))
torch.save(model, "model_separated.ckpt")
valid_epoch_alt(model, valid_loader)
