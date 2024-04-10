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

class ConvDownSample(nn.Module):
    def __init__(self, input_shape=150, hidden_dim=4, output_dim=100):
        super(ConvDownSample, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_shape = input_shape
        self.conv1_1 = nn.Conv2d(3, hidden_dim, kernel_size = 5, padding = "same")
        self.conv2_1 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size = 5, stride = 5)
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv1_2 = nn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size = 3, padding = "same")
        self.conv2_2 = nn.Conv2d(hidden_dim*2, hidden_dim * 4, kernel_size = 3, stride = 3)
        self.conv1_3 = nn.Conv2d(hidden_dim*4, hidden_dim*4, kernel_size = 3, padding = "same")
        self.conv2_3 = nn.Conv2d(hidden_dim*4, hidden_dim * 8, kernel_size = 3, stride = 2)
        self.conv1_4 = nn.Conv2d(hidden_dim*8, hidden_dim*8, kernel_size = 3, padding = "same")
        self.conv2_4 = nn.Conv2d(hidden_dim*8, output_dim, kernel_size = 3, padding = "same")

    def forward(self, input):
        x, t = input
        # t_embed = self.t_linear(t)
        x = self.lrelu(self.conv1_1(x))
        x = self.lrelu(self.conv2_1(x))
        x = x + self.lrelu(self.conv1_2(x))
        x = self.lrelu(self.conv2_2(x))
        x = x + self.lrelu(self.conv1_3(x))
        x = self.lrelu(self.conv2_3(x))
        x = x + self.lrelu(self.conv1_4(x))
        x = self.conv2_4(x)
        return(x)
    
class ConvLayer(nn.Module):
    def __init__(self, input_shape=5, hidden_dim=4, output_dim=100):
        super(ConvLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_shape = input_shape
        self.conv1_1 = nn.Conv2d(3, hidden_dim, kernel_size = 3, padding = "same")
        self.conv2_1 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size = 3,padding = "same")
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv1_2 = nn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size = 3, padding = "same")
        self.conv2_2 = nn.Conv2d(hidden_dim*2, hidden_dim * 4, kernel_size = 3, padding = "same")
        self.conv1_3 = nn.Conv2d(hidden_dim*4, hidden_dim*4, kernel_size = 3, padding = "same")
        self.conv2_3 = nn.Conv2d(hidden_dim*4, hidden_dim * 8, kernel_size = 3, padding = "same")
        self.conv1_4 = nn.Conv2d(hidden_dim*8, hidden_dim*8, kernel_size = 3, padding = "same")
        self.conv2_4 = nn.Conv2d(hidden_dim*8, output_dim, kernel_size = 3, padding = "same")

    def forward(self, input):
        x, t = input
        # t_embed = self.t_linear(t)
        x = self.lrelu(self.conv1_1(x))
        x = self.lrelu(self.conv2_1(x))
        x = x + self.lrelu(self.conv1_2(x))
        x = self.lrelu(self.conv2_2(x))
        x = x + self.lrelu(self.conv1_3(x))
        x = self.lrelu(self.conv2_3(x))
        x = x + self.lrelu(self.conv1_4(x))
        x = self.conv2_4(x)
        return(x)
    
class ConvUpSample(nn.Module):
    def __init__(self, input_shape=5, hidden_dim=4, output_dim=100):
        super(ConvUpSample, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_shape = input_shape
        self.conv1_1 = nn.Conv2d(output_dim, hidden_dim*8, kernel_size = 3, padding = "same")
        self.conv2_1 = nn.Conv2d(hidden_dim*8, hidden_dim *8, kernel_size = 3,padding = "same")
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv1_2 = nn.Conv2d(hidden_dim*8, hidden_dim*4, kernel_size = 3, padding = "same")
        self.conv2_2 = nn.Conv2d(hidden_dim*4, hidden_dim * 4, kernel_size = 3, padding = "same")
        self.conv1_3 = nn.Conv2d(hidden_dim*4, hidden_dim*2, kernel_size = 3, padding = "same")
        self.conv2_3 = nn.Conv2d(hidden_dim*2, hidden_dim * 2, kernel_size = 3, padding = "same")
        self.conv1_4 = nn.Conv2d(hidden_dim*2, hidden_dim, kernel_size = 3, padding = "same")
        self.conv2_4 = nn.Conv2d(hidden_dim, 3, kernel_size = 3, padding = "same")

    def forward(self, input):
        x, t = input
        # t_embed = self.t_linear(t)
        x = self.lrelu(self.conv1_1(x))
        x = x+self.lrelu(self.conv2_1(x))
        x = nn.functional.interpolate(x, [self.input_shape//(5*3), self.input_shape//(5*3)], mode='bilinear')
        x = self.lrelu(self.conv1_2(x))
        x = x+self.lrelu(self.conv2_2(x))
        x = nn.functional.interpolate(x, [self.input_shape//(5), self.input_shape//(5)], mode='bilinear')
        x = self.lrelu(self.conv1_3(x))
        x = x+self.lrelu(self.conv2_3(x))
        x = nn.functional.interpolate(x, [self.input_shape, self.input_shape], mode='bilinear')
        x = self.lrelu(self.conv1_4(x))
        x = self.conv2_4(x)
        return(x)
    
class VAE(nn.Module):
    def __init__(self, input_shape=150, sample_shape = 5, hidden_dim=4, latent_dim=10):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        # encoder
        self.encoder = ConvDownSample(input_shape, hidden_dim, latent_dim*2)
        
        # encoder
        self.encoder_sample = ConvLayer(input_shape, hidden_dim, latent_dim*2)
    
        # decoder
        self.decoder = ConvUpSample(input_shape, hidden_dim, latent_dim)
     
    def encode(self, x):
        x = self.encoder(x)
        mean = x[:, :self.latent_dim, :, :]
        logvar = x[:, self.latent_dim:, :, :]
        return mean, logvar
    
    def encode_sample(self, x):
        x = self.encoder_sample(x)
        mean = x[:, :self.latent_dim, :, :]
        logvar = x[:, self.latent_dim:, :, :]
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
# data_x, data_y, data_t = data_x[:5000], data_y[:5000], data_t[:5000]
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
model = VAE(150, 0, 4, 16)
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
            t = t/21/5000
            target = x
            optimizer.zero_grad()
            x_hat, mean, log_var = model((target, t))
            x_hat_sample, mean_sample, log_var_sample = model.forward_sample((target[:, :, ::30, ::30], t))
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
            t = t/21/5000
            target = x
            optimizer.zero_grad()
            x_hat, mean, log_var = model((target, t))
            x_hat_sample, mean_sample, log_var_sample = model.forward_sample((target[:, :, ::30, ::30], t))
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
    train_epoch(model, optimizer, train_loader)
    valid_epoch(model, valid_loader)
torch.save(model, "model_rom.ckpt")
