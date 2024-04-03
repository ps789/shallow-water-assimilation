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

class Triple_Layer_with_Embedding(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=100, output_dim=100):
        super(Triple_Layer_with_Embedding, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.lrelu = nn.LeakyReLU(0.2)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)
        self.t_linear = nn.Linear(1, hidden_dim)

    def forward(self, input):
        x, t = input
        t_embed = self.t_linear(t)
        x = self.lrelu(self.linear1(x) + t_embed)
        x = self.lrelu(self.linear2(x) + t_embed)
        x = self.linear3(x)
        return(x)
    
class VAE(nn.Module):
    def __init__(self, input_dim=100, sample_dim= 10, hidden_dim=100, latent_dim=10):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        # encoder
        self.encoder = Triple_Layer_with_Embedding(input_dim, hidden_dim, latent_dim*2)
        
        # encoder
        self.encoder_sample = Triple_Layer_with_Embedding(sample_dim, hidden_dim, latent_dim*2)
    
        # decoder
        self.decoder = Triple_Layer_with_Embedding(latent_dim, hidden_dim, input_dim)
     
    def encode(self, x):
        x = self.encoder(x)
        mean = x[:, :self.latent_dim]
        logvar = x[:, self.latent_dim:]
        return mean, logvar
    
    def encode_sample(self, x):
        x = self.encoder_sample(x)
        mean = x[:, :self.latent_dim]
        logvar = x[:, self.latent_dim:]
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)      
        z = mean + var*epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, input):
        _, t = input
        mean, logvar = self.encode(input)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode((z, t))
        return x_hat, mean, logvar
    def forward_sample(self, input):
        _, t = input
        mean, logvar = self.encode_sample(input)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode((z, t))
        return x_hat, mean, logvar
    
mseloss = torch.nn.MSELoss(reduction='sum')
def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = mseloss(x_hat, x)
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss, reproduction_loss + KLD

X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(data_x, data_y, data_t, test_size = 0.2, shuffle = False)#, random_state = 42)

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
model = VAE(150*150*3, 3*150*150//25, 784, 128)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_dataset = CustomDataset(X_train, y_train, t_train, device)
valid_dataset = CustomDataset(X_test, y_test, t_test, device)
def train_epoch(model, optimizer, train_loader):
    losses = []
    model.train()
    with tqdm(total=len(train_loader), desc=f"Train {epoch}: ") as pbar:
        for i, value in enumerate(train_loader):
            x, y, t = value
            target = x.view(x.shape[0], -1)
            optimizer.zero_grad()
            x_hat, mean, log_var = model((target, t))
            x_hat_sample, mean_sample, log_var_sample = model.forward_sample((target[:, ::25], t))
            target_reconstruction, loss_target = loss_function(target, x_hat, mean, log_var)
            sample_reconstruction, loss_sample = loss_function(target, x_hat_sample, mean_sample, log_var_sample)
            loss = loss_target + loss_sample
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            # scheduler.step()
            pbar.update(1)
            pbar.set_postfix_str(
                f"Loss: {loss:.3f} ({np.mean(losses):.3f}))")
    return np.mean(losses)
def valid_epoch(model, valid_loader):
    losses = []
    model.eval()
    with tqdm(total=len(valid_loader), desc=f"Valid {epoch}: ") as pbar:
        for i, value in enumerate(valid_loader):
            x, y, t = value
            target = x.view(x.shape[0], -1)
            optimizer.zero_grad()
            x_hat, mean, log_var = model((target, t))
            x_hat_sample, mean_sample, log_var_sample = model.forward_sample((target[:, ::25], t))
            target_reconstruction, loss_target = loss_function(target, x_hat, mean, log_var)
            sample_reconstruction, loss_sample = loss_function(target, x_hat_sample, mean_sample, log_var_sample)
            loss = loss_target + loss_sample

            loss_val = loss.item()
            losses.append(loss_val)
            # scheduler.step()
            pbar.update(1)
            pbar.set_postfix_str(
                f"Loss: {loss_val:.3f} ({np.mean(losses):.3f}))")
    return np.mean(losses)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False)
for epoch in range(200):
    print('EPOCH {}:'.format(epoch + 1))
    print(train_epoch(model, optimizer, train_loader))
    print(valid_epoch(model, valid_loader))
torch.save(model, "model_rom.ckpt")
