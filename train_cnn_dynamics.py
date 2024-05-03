import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import time
data_x = np.load("./sequential_data/x_data_sequential.npy", allow_pickle = True)
# data_x = np.load("./sequential_data/x_data_sequential_0.npy", allow_pickle = True)

class ConvDownSample(nn.Module):
    def __init__(self, input_shape=150, hidden_dim=4, output_dim=100):
        super(ConvDownSample, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_shape = input_shape
        self.conv1_1 = nn.Conv2d(3, hidden_dim, kernel_size = 5, padding = "same")
        self.conv2_1 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size = 5, stride = 5, padding = 2)
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv1_2 = nn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size = 3, padding = "same")
        self.conv2_2 = nn.Conv2d(hidden_dim*2, hidden_dim * 4, kernel_size = 3, stride = 3, padding = 1)
        self.conv1_3 = nn.Conv2d(hidden_dim*4, hidden_dim*4, kernel_size = 3, padding = "same")
        self.conv2_3 = nn.Conv2d(hidden_dim*4, hidden_dim * 8, kernel_size = 3, stride = 2, padding = 1)
        self.conv1_4 = nn.Conv2d(hidden_dim*8, hidden_dim*8, kernel_size = 3, padding = "same")
        self.conv2_4 = nn.Conv2d(hidden_dim*8, output_dim, kernel_size = 3, padding = "same")

    def forward(self, input):
        x = input
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
        x = input
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
        x = input
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
    
class ConvLayer_2(nn.Module):
    def __init__(self, input_shape=5, hidden_dim=4, output_dim=100):
        super(ConvLayer_2, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_shape = input_shape
        self.conv1_1 = nn.Conv2d(3, hidden_dim, kernel_size = 3, padding = "same")
        self.conv2_1 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size = 3,padding = "same")
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv1_2 = nn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size = 3, padding = "same")
        self.conv2_2 = nn.Conv2d(hidden_dim*2, hidden_dim * 4, kernel_size = 3, padding = "same")
        self.conv1_3 = nn.Conv2d(hidden_dim*4, hidden_dim*4, kernel_size = 3, padding = "same")
        self.conv2_3 = nn.Conv2d(hidden_dim*4, hidden_dim * 8, kernel_size = 3, stride = 2, padding = 1)
        self.conv1_4 = nn.Conv2d(hidden_dim*8, hidden_dim*8, kernel_size = 3, padding = "same")
        self.conv2_4 = nn.Conv2d(hidden_dim*8, output_dim, kernel_size = 3, padding = "same")

    def forward(self, input):
        x = input
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
    
    
class VAE(nn.Module):
    def __init__(self, input_shape=150, sample_shape = 5, hidden_dim=4, latent_dim=10):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        # encoder
        self.encoder = ConvDownSample(input_shape, hidden_dim, latent_dim*2)
        
        # encoder
        self.encoder_sample = ConvLayer_2(input_shape, hidden_dim, latent_dim*2)
    
        # decoder
        self.decoder = ConvUpSample(input_shape, hidden_dim, latent_dim)
        
        self.lstm = nn.LSTM(input_size=5*5*latent_dim*2, hidden_size=256, num_layers=4, batch_first=True)
        self.linear = nn.Linear(256, 5*5*latent_dim*2)
     
    def encode(self, x):
        x = self.encoder(x)
        return x
    
    def encode_sample(self, x):
        x = self.encoder_sample(x)
        return x
    
    def forward_latent(self, mean_logvar):
        mean_logvar_sequence = mean_logvar.reshape((mean_logvar.shape[0], mean_logvar.shape[1], -1))
        x, (h, c) = self.lstm(mean_logvar_sequence)
        x = self.linear(x.reshape(-1, 256)).reshape(mean_logvar.shape)
        return x
    
    def split(self, x):
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
        input_reshaped = input.view((input.shape[0] * input.shape[1], input.shape[2], input.shape[3], input.shape[4]))
        mean_logvar = self.encode(input_reshaped)
        #batch* seq_len, 8, 5, 5
        mean_logvar_sequence = mean_logvar.reshape((input.shape[0], input.shape[1], -1))
        #batch, seq_len, 8*5*5
        lstm_out, (h, c) = self.lstm(mean_logvar_sequence)
        #batch, seq_len, 256
        mean_logvar_sequence = self.linear(lstm_out.reshape(-1, 256)).reshape(mean_logvar_sequence.shape)
        #batch*seq_len, 256 -> batch*seq_len, 200 -> batch, seq_len, 8*5*5
        mean, logvar = self.split(mean_logvar)
        mean_sequence, logvar_sequence = self.split(mean_logvar_sequence.reshape((input.shape[0]* input.shape[1], self.latent_dim*2, 5, 5)))
        z = self.reparameterization(mean, logvar)
        z_sequence = self.reparameterization(mean_sequence, logvar_sequence)
        x_hat = self.decode(z).view((input.shape[0], input.shape[1], input.shape[2], input.shape[3], input.shape[4]))
        x_hat_sequence = self.decode(z_sequence).view((input.shape[0], input.shape[1], input.shape[2], input.shape[3], input.shape[4]))
        return x_hat, mean.view((input.shape[0], input.shape[1], self.latent_dim, 5, 5)), logvar.view((input.shape[0], input.shape[1], self.latent_dim, 5, 5)), x_hat_sequence, mean_sequence.view((input.shape[0], input.shape[1], self.latent_dim, 5, 5)), logvar_sequence.view((input.shape[0], input.shape[1], self.latent_dim, 5, 5))
    
    def forward_sample(self, input):
        input_reshaped = input.view((input.shape[0] * input.shape[1], input.shape[2], input.shape[3], input.shape[4]))
        mean_logvar = self.encode_sample(input_reshaped)
        #batch* seq_len, 8, 5, 5
        mean_logvar_sequence = mean_logvar.reshape((input.shape[0], input.shape[1], -1))
        #batch, seq_len, 8*5*5
        lstm_out, (h, c) = self.lstm(mean_logvar_sequence)
        #batch, seq_len, 256
        mean_logvar_sequence = self.linear(lstm_out.reshape(-1, 256)).reshape(mean_logvar_sequence.shape)
        #batch*seq_len, 256 -> batch*seq_len, 200 -> batch, seq_len, 8*5*5
        mean, logvar = self.split(mean_logvar)
        mean_sequence, logvar_sequence = self.split(mean_logvar_sequence.reshape((input.shape[0]* input.shape[1], self.latent_dim*2, 5, 5)))
        z = self.reparameterization(mean, logvar)
        z_sequence = self.reparameterization(mean_sequence, logvar_sequence)
        x_hat = self.decode(z).view((input.shape[0], input.shape[1], input.shape[2], 150, 150))
        x_hat_sequence = self.decode(z_sequence).view((input.shape[0], input.shape[1], input.shape[2], 150, 150))
        return x_hat, mean.view((input.shape[0], input.shape[1], self.latent_dim, 5, 5)), logvar.view((input.shape[0], input.shape[1], self.latent_dim, 5, 5)), x_hat_sequence, mean_sequence.view((input.shape[0], input.shape[1], self.latent_dim, 5, 5)), logvar_sequence.view((input.shape[0], input.shape[1], self.latent_dim, 5, 5))

mseloss = torch.nn.MSELoss(reduction='sum')
def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = mseloss(x_hat, x)
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD

def kld(mean1, log_var1, mean2, log_var2):
    # return torch.sum((mean1 - mean2).pow(2))
    return - 0.5 * torch.sum(1+ log_var1 - log_var2 - (mean1 - mean2).pow(2)/log_var2.exp() - log_var1.exp()/log_var2.exp())
# data_x, data_y, data_t = data_x[:5000], data_y[:5000], data_t[:5000]
X_train, X_test = train_test_split(data_x, test_size = 0.2, shuffle = False)#, random_state = 42)

class CustomDataset(Dataset):

    def __init__(self, x_data, device):
        self.device = device
        self.x_data = torch.Tensor(x_data)
    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):

        return self.x_data[idx].to(self.device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, flush = True)
model = VAE(150, 0, 4, 4)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_dataset = CustomDataset(X_train, device)
valid_dataset = CustomDataset(X_test, device)
def train_epoch(model, optimizer, train_loader):
    losses = []
    model.train()
    with tqdm(total=len(train_loader), desc=f"Train {epoch}: ") as pbar:
        for i, value in enumerate(train_loader):
            x = value
            target = x
            optimizer.zero_grad()
            x_hat, mean, log_var, x_hat_sequence, mean_sequence, log_var_sequence = model(target)
            x_hat_sample, mean_sample, log_var_sample, x_hat_sequence_sample, mean_sequence_sample, log_var_sequence_sample = model.forward_sample(target[:, :, :, ::15, ::15])
            loss_target = loss_function(target, x_hat, mean, log_var)
            loss_target_sequence = loss_function(target[:, 1:, :, :, :], x_hat_sequence[:, :-1, :, :, :], mean_sequence[:, :-1, :, :, :], log_var_sequence[:, :-1, :, :, :])
            # sample_reconstruction, loss_sample = loss_function(target, x_hat_sample, mean_sample, log_var_sample)
            loss_full = loss_target  + loss_target_sequence + mseloss(mean_sequence[:, :-1, :, :, :], mean[:, 1:, :, :, :]) + mseloss(log_var_sequence[:, :-1, :, :, :], log_var[:, 1:, :, :, :]) # + loss_sample# + kld(mean_sample, log_var_sample, mean, log_var) + kld(mean, log_var, mean_sample, log_var_sample)
            
            loss_sample = loss_function(target, x_hat_sample, mean_sample, log_var_sample) +  loss_function(target[:, 1:, :, :, :], x_hat_sequence_sample[:, :-1, :, :, :], mean_sequence_sample[:, :-1, :, :, :], log_var_sequence_sample[:, :-1, :, :, :]) + mseloss(mean_sequence_sample[:, :-1, :, :, :], mean[:, 1:, :, :, :]) + mseloss(log_var_sequence_sample[:, :-1, :, :, :], log_var[:, 1:, :, :, :])
            loss = loss_full + loss_sample
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
            x = value
            target = x
            optimizer.zero_grad()
            with torch.no_grad():
                x_hat, mean, log_var, x_hat_sequence, mean_sequence, log_var_sequence = model(target)
                x_hat_sample, mean_sample, log_var_sample, x_hat_sequence_sample, mean_sequence_sample, log_var_sequence_sample = model.forward_sample(target[:, :, :, ::15, ::15])
                # x_hat_sample, mean_sample, log_var_sample = model.forward_sample((target[:, :, ::30, ::30], t))
                loss_target = loss_function(target, x_hat, mean, log_var)
                loss_target_sequence = loss_function(target[:, 1:, :, :, :], x_hat_sequence[:, :-1, :, :, :], mean_sequence[:, :-1, :, :, :], log_var_sequence[:, :-1, :, :, :])
                # sample_reconstruction, loss_sample = loss_function(target, x_hat_sample, mean_sample, log_var_sample)
                loss_full = loss_target  + loss_target_sequence + mseloss(mean_sequence[:, :-1, :, :, :], mean[:, 1:, :, :, :]) + mseloss(log_var_sequence[:, :-1, :, :, :], log_var[:, 1:, :, :, :]) # + loss_sample# + kld(mean_sample, log_var_sample, mean, log_var) + kld(mean, log_var, mean_sample, log_var_sample)
                
                loss_sample = loss_function(target, x_hat_sample, mean_sample, log_var_sample) +  loss_function(target[:, 1:, :, :, :], x_hat_sequence_sample[:, :-1, :, :, :], mean_sequence_sample[:, :-1, :, :, :], log_var_sequence_sample[:, :-1, :, :, :]) + mseloss(mean_sequence_sample[:, :-1, :, :, :], mean[:, 1:, :, :, :]) + mseloss(log_var_sequence_sample[:, :-1, :, :, :], log_var[:, 1:, :, :, :])
                loss = loss_full + loss_sample

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
    if (epoch + 1)%50 == 0:
        torch.save(model, f"model_cnn_dynamics_15_{epoch}.ckpt")
