
"""Script that solves that solves the 2D shallow water equations using finite
differences where the momentum equations are taken to be linear, but the
continuity equation is solved in its nonlinear form. The model supports turning
on/off various terms, but in its mst complete form, the model solves the following
set of eqations:

    du/dt - fv = -g*d(eta)/dx + tau_x/(rho_0*H)- kappa*u
    dv/dt + fu = -g*d(eta)/dy + tau_y/(rho_0*H)- kappa*v
    d(eta)/dt + d((eta + H)*u)/dx + d((eta + H)*u)/dy = sigma - w

where f = f_0 + beta*y can be the full latitude varying coriolis parameter.
For the momentum equations, an ordinary forward-in-time centered-in-space
scheme is used. However, the coriolis terms is not so trivial, and thus, one
first finds a predictor for u, v and then a corrected value is computed in
order to include the coriolis terms. In the continuity equation, it's used a
forward difference for the time derivative and an upwind scheme for the non-
linear terms. The model is stable under the CFL condition of

    dt <= min(dx, dy)/sqrt(g*H)    and    alpha << 1 (if coriolis is used)

where dx, dy is the grid spacing in the x- and y-direction respectively, g is
the acceleration of gravity and H is the resting depth of the fluid."""

import time
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import viz_tools
import torch
import torch.nn as nn
torch.set_default_dtype(torch.float32) # half precision
device = 'cuda'

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

    
mseloss = torch.nn.MSELoss(reduction='mean')
def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = mseloss(x_hat, x)
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss, reproduction_loss + KLD
# ==================================================================================
# ================================ Parameter stuff =================================
# ==================================================================================
# --------------- Physical prameters ---------------
L_x = 1E+6              # Length of domain in x-direction
L_y = 1E+6              # Length of domain in y-direction
g = 9.81                 # Acceleration of gravity [m/s^2]
H = 100                # Depth of fluid [m]
f_0 = 1E-4              # Fixed part ofcoriolis parameter [1/s]
beta = 2E-11            # gradient of coriolis parameter [1/ms]
rho_0 = 1024.0          # Density of fluid [kg/m^3)]
tau_0 = 0.1             # Amplitude of wind stress [kg/ms^2]
use_coriolis = True     # True if you want coriolis force
use_friction = False     # True if you want bottom friction
use_wind = False        # True if you want wind stress
use_beta = True         # True if you want variation in coriolis
use_source = False       # True if you want mass source into the domain
use_sink = False       # True if you want mass sink out of the domain
param_string = "\n================================================================"
param_string += "\nuse_coriolis = {}\nuse_beta = {}".format(use_coriolis, use_beta)
param_string += "\nuse_friction = {}\nuse_wind = {}".format(use_friction, use_wind)
param_string += "\nuse_source = {}\nuse_sink = {}".format(use_source, use_sink)
param_string += "\ng = {:g}\nH = {:g}".format(g, H)

# --------------- Computational prameters ---------------
N_x = 150                            # Number of grid points in x-direction
N_y = 150                            # Number of grid points in y-direction

# model = VAE(150*150*3, 3*150*150//25, 784, 100).to(device)
# temp_model = torch.load("model_rom.ckpt")
# model.load_state_dict(temp_model.state_dict())
latent_channels = 4
cnn_model = VAE(150, 0, 4, latent_channels).to(device)
temp_model = torch.load("./model_cnn_dynamics_15_199.ckpt")
cnn_model.load_state_dict(temp_model.state_dict())

dx = L_x/(N_x - 1)                   # Grid spacing in x-direction
dy = L_y/(N_y - 1)                   # Grid spacing in y-direction
dt = 0.1*min(dx, dy)/np.sqrt(g*H)    # Time step (defined from the CFL condition)
time_step = 1                        # For counting time loop steps
max_time_step = 5000                 # Total number of time steps in simulation
anim_interval = 20
x = np.linspace(-L_x/2, L_x/2, N_x)  # Array with x-points
y = np.linspace(-L_y/2, L_y/2, N_y)  # Array with y-points
X, Y = np.meshgrid(x, y)             # Meshgrid for plotting
X = np.transpose(X)                  # To get plots right
Y = np.transpose(Y)                  # To get plots right
param_string += "\ndx = {:.2f} km\ndy = {:.2f} km\ndt = {:.2f} s".format(dx, dy, dt)

# Define friction array if friction is enabled.
if (use_friction is True):
    kappa_0 = 1/(5*24*3600)
    kappa = np.ones((N_x, N_y))*kappa_0
    #kappa[0, :] = kappa_0
    #kappa[-1, :] = kappa_0
    #kappa[:, 0] = kappa_0
    #kappa[:, -1] = kappa_0
    #kappa[:int(N_x/15), :] = 0
    #kappa[int(14*N_x/15)+1:, :] = 0
    #kappa[:, :int(N_y/15)] = 0
    #kappa[:, int(14*N_y/15)+1:] = 0
    #kappa[int(N_x/15):int(2*N_x/15), int(N_y/15):int(14*N_y/15)+1] = 0
    #kappa[int(N_x/15):int(14*N_x/15)+1, int(N_y/15):int(2*N_y/15)] = 0
    #kappa[int(13*N_x/15)+1:int(14*N_x/15)+1, int(N_y/15):int(14*N_y/15)+1] = 0
    #kappa[int(N_x/15):int(14*N_x/15)+1, int(13*N_y/15)+1:int(14*N_y/15)+1] = 0
    param_string += "\nkappa = {:g}\nkappa/beta = {:g} km".format(kappa_0, kappa_0/(beta*1000))

# Define wind stress arrays if wind is enabled.
if (use_wind is True):
    tau_x = -tau_0*np.cos(np.pi*y/L_y)*0
    tau_y = np.zeros((1, len(x)))
    param_string += "\ntau_0 = {:g}\nrho_0 = {:g} km".format(tau_0, rho_0)

# Define coriolis array if coriolis is enabled.
if (use_coriolis is True):
    if (use_beta is True):
        f = f_0 + beta*y        # Varying coriolis parameter
        L_R = np.sqrt(g*H)/f_0  # Rossby deformation radius
        c_R = beta*g*H/f_0**2   # Long Rossby wave speed
    else:
        f = f_0*np.ones(len(y))                 # Constant coriolis parameter

    alpha = dt*f                # Parameter needed for coriolis scheme
    beta_c = alpha**2/4         # Parameter needed for coriolis scheme

    param_string += "\nf_0 = {:g}".format(f_0)
    param_string += "\nMax alpha = {:g}\n".format(alpha.max())
    param_string += "\nRossby radius: {:.1f} km".format(L_R/1000)
    param_string += "\nRossby number: {:g}".format(np.sqrt(g*H)/(f_0*L_x))
    param_string += "\nLong Rossby wave speed: {:.3f} m/s".format(c_R)
    param_string += "\nLong Rossby transit time: {:.2f} days".format(L_x/(c_R*24*3600))
    param_string += "\n================================================================\n"

# Define source array if source is enabled.
if (use_source):
    sigma = np.zeros((N_x, N_y))
    sigma = 0.0001*np.exp(-((X-L_x/2)**2/(2*(1E+5)**2) + (Y-L_y/2)**2/(2*(1E+5)**2)))
    
# Define source array if source is enabled.
if (use_sink is True):
    w = np.ones((N_x, N_y))*sigma.sum()/(N_x*N_y)

# Write all parameters out to file.
with open("param_output.txt", "w") as output_file:
    output_file.write(param_string)

print(param_string)     # Also print parameters to screen
# ============================= Parameter stuff done ===============================

# ==================================================================================
# ==================== Allocating arrays and initial conditions ====================
# ==================================================================================
u_n = np.zeros((N_x, N_y))      # To hold u at current time step
u_np1 = np.zeros((N_x, N_y))    # To hold u at next time step
v_n = np.zeros((N_x, N_y))      # To hold v at current time step
v_np1 = np.zeros((N_x, N_y))    # To hold v at enxt time step
eta_n = np.zeros((N_x, N_y))    # To hold eta at current time step
eta_np1 = np.zeros((N_x, N_y))  # To hold eta at next time step

# Temporary variables (each time step) for upwind scheme in eta equation
h_e = np.zeros((N_x, N_y))
h_w = np.zeros((N_x, N_y))
h_n = np.zeros((N_x, N_y))
h_s = np.zeros((N_x, N_y))
uhwe = np.zeros((N_x, N_y))
vhns = np.zeros((N_x, N_y))

# Initial conditions for u and v.
u_n[:, :] = 0.0             # Initial condition for u
v_n[:, :] = 0.0             # Initial condition for u
u_n[-1, :] = 0.0            # Ensuring initial u satisfy BC
v_n[:, -1] = 0.0            # Ensuring initial v satisfy BC

# Initial condition for eta.
#eta_n[:, :] = np.sin(4*np.pi*X/L_y) + np.sin(4*np.pi*Y/L_y)
#eta_n = np.exp(-((X-0)**2/(2*(L_R)**2) + (Y-0)**2/(2*(L_R)**2)))
eta_n = np.exp(-((X-L_x/2.7)**2/(2*(0.05E+6)**2) + (Y-L_y/4)**2/(2*(0.05E+6)**2)))
#eta_n[int(3*N_x/8):int(5*N_x/8),int(3*N_y/8):int(5*N_y/8)] = 1.0
#eta_n[int(6*N_x/8):int(7*N_x/8),int(6*N_y/8):int(7*N_y/8)] = 1.0
#eta_n[int(3*N_x/8):int(5*N_x/8), int(13*N_y/14):] = 1.0
#eta_n[:, :] = 0.0

#viz_tools.surface_plot3D(X, Y, eta_n, (X.min(), X.max()), (Y.min(), Y.max()), (eta_n.min(), eta_n.max()))

# Sampling variables.
eta_list = list(); u_list = list(); v_list = list()         # Lists to contain eta and u,v for animation
eta_list_decoded = list(); u_list_decoded = list(); v_list_decoded = list()         # Lists to contain eta and u,v for animation
eta_list_decoded_sparse = list(); u_list_decoded_sparse = list(); v_list_decoded_sparse = list()         # Lists to contain eta and u,v for animation
hm_sample = list(); ts_sample = list(); t_sample = list()   # Lists for Hovmuller and time series
hm_sample.append(eta_n[:, int(N_y/2)])                      # Sample initial eta in middle of domain
ts_sample.append(eta_n[int(N_x/2), int(N_y/2)])             # Sample initial eta at center of domain
t_sample.append(0.0)                                        # Add initial time to t-samples
sample_interval = 1000                                      # How often to sample for time series
# =============== Done with setting up arrays and initial conditions ===============

t_0 = time.perf_counter()  # For timing the computation loop

# ==================================================================================
# ========================= Main time loop for simulation ==========================
# ==================================================================================
while (time_step < max_time_step):
    # ------------ Computing values for u and v at next time step --------------
    u_np1[:-1, :] = u_n[:-1, :] - g*dt/dx*(eta_n[1:, :] - eta_n[:-1, :])
    v_np1[:, :-1] = v_n[:, :-1] - g*dt/dy*(eta_n[:, 1:] - eta_n[:, :-1])

    # Add friction if enabled.
    if (use_friction is True):
        u_np1[:-1, :] -= dt*kappa[:-1, :]*u_n[:-1, :]
        v_np1[:-1, :] -= dt*kappa[:-1, :]*v_n[:-1, :]

    # Add wind stress if enabled.
    if (use_wind is True):
        u_np1[:-1, :] += dt*tau_x[:]/(rho_0*H)
        v_np1[:-1, :] += dt*tau_y[:]/(rho_0*H)

    # Use a corrector method to add coriolis if it's enabled.
    if (use_coriolis is True):
        u_np1[:, :] = (u_np1[:, :] - beta_c*u_n[:, :] + alpha*v_n[:, :])/(1 + beta_c)
        v_np1[:, :] = (v_np1[:, :] - beta_c*v_n[:, :] - alpha*u_n[:, :])/(1 + beta_c)
    
    v_np1[:, -1] = 0.0      # Northern boundary condition
    u_np1[-1, :] = 0.0      # Eastern boundary condition
    # -------------------------- Done with u and v -----------------------------

    # --- Computing arrays needed for the upwind scheme in the eta equation.----
    h_e[:-1, :] = np.where(u_np1[:-1, :] > 0, eta_n[:-1, :] + H, eta_n[1:, :] + H)
    h_e[-1, :] = eta_n[-1, :] + H

    h_w[0, :] = eta_n[0, :] + H
    h_w[1:, :] = np.where(u_np1[:-1, :] > 0, eta_n[:-1, :] + H, eta_n[1:, :] + H)

    h_n[:, :-1] = np.where(v_np1[:, :-1] > 0, eta_n[:, :-1] + H, eta_n[:, 1:] + H)
    h_n[:, -1] = eta_n[:, -1] + H

    h_s[:, 0] = eta_n[:, 0] + H
    h_s[:, 1:] = np.where(v_np1[:, :-1] > 0, eta_n[:, :-1] + H, eta_n[:, 1:] + H)

    uhwe[0, :] = u_np1[0, :]*h_e[0, :]
    uhwe[1:, :] = u_np1[1:, :]*h_e[1:, :] - u_np1[:-1, :]*h_w[1:, :]

    vhns[:, 0] = v_np1[:, 0]*h_n[:, 0]
    vhns[:, 1:] = v_np1[:, 1:]*h_n[:, 1:] - v_np1[:, :-1]*h_s[:, 1:]
    # ------------------------- Upwind computations done -------------------------

    # ----------------- Computing eta values at next time step -------------------
    eta_np1[:, :] = eta_n[:, :] - dt*(uhwe[:, :]/dx + vhns[:, :]/dy)    # Without source/sink

    # Add source term if enabled.
    if (use_source is True):
        eta_np1[:, :] += dt*sigma

    # Add sink term if enabled.
    if (use_sink is True):
        eta_np1[:, :] -= dt*w
    # ----------------------------- Done with eta --------------------------------

    u_n = np.copy(u_np1)        # Update u for next iteration
    v_n = np.copy(v_np1)        # Update v for next iteration
    eta_n = np.copy(eta_np1)    # Update eta for next iteration

    time_step += 1

    # Samples for Hovmuller diagram and spectrum every sample_interval time step.
    if (time_step % sample_interval == 0):
        hm_sample.append(eta_n[:, int(N_y/2)])              # Sample middle of domain for Hovmuller
        ts_sample.append(eta_n[int(N_x/2), int(N_y/2)])     # Sample center point for spectrum
        t_sample.append(time_step*dt)                       # Keep track of sample times.

    # Store eta and (u, v) every anin_interval time step for animations.
    if (time_step % anim_interval == 0):
        print("Time: \t{:.2f} hours".format(time_step*dt/3600))
        print("Step: \t{} / {}".format(time_step, max_time_step))
        print("Mass: \t{}\n".format(np.sum(eta_n)))
        u_list.append(u_n)
        v_list.append(v_n)
        eta_list.append(eta_n)

# # ==================Perturbed==================================================#

rmse_list_perturbed = []
# ==================================================================================
# ==================== Allocating arrays and initial conditions ====================
# ==================================================================================
u_n = np.zeros((N_x, N_y))      # To hold u at current time step
u_np1 = np.zeros((N_x, N_y))    # To hold u at next time step
v_n = np.zeros((N_x, N_y))      # To hold v at current time step
v_np1 = np.zeros((N_x, N_y))    # To hold v at enxt time step
eta_n = np.zeros((N_x, N_y))    # To hold eta at current time step
eta_np1 = np.zeros((N_x, N_y))  # To hold eta at next time step

# Temporary variables (each time step) for upwind scheme in eta equation
h_e = np.zeros((N_x, N_y))
h_w = np.zeros((N_x, N_y))
h_n = np.zeros((N_x, N_y))
h_s = np.zeros((N_x, N_y))
uhwe = np.zeros((N_x, N_y))
vhns = np.zeros((N_x, N_y))

# Initial conditions for u and v.
u_n[:, :] = 0.0             # Initial condition for u
v_n[:, :] = 0.0             # Initial condition for u
u_n[-1, :] = 0.0            # Ensuring initial u satisfy BC
v_n[:, -1] = 0.0            # Ensuring initial v satisfy BC

# Initial condition for eta.
#eta_n[:, :] = np.sin(4*np.pi*X/L_y) + np.sin(4*np.pi*Y/L_y)
#eta_n = np.exp(-((X-0)**2/(2*(L_R)**2) + (Y-0)**2/(2*(L_R)**2)))
eta_n = np.exp(-((X-L_x/2)**2/(2*(0.05E+6)**2) + (Y-L_y/3)**2/(2*(0.05E+6)**2)))
#eta_n[int(3*N_x/8):int(5*N_x/8),int(3*N_y/8):int(5*N_y/8)] = 1.0
#eta_n[int(6*N_x/8):int(7*N_x/8),int(6*N_y/8):int(7*N_y/8)] = 1.0
#eta_n[int(3*N_x/8):int(5*N_x/8), int(13*N_y/14):] = 1.0
#eta_n[:, :] = 0.0

#viz_tools.surface_plot3D(X, Y, eta_n, (X.min(), X.max()), (Y.min(), Y.max()), (eta_n.min(), eta_n.max()))

# Sampling variables.
time_step = 1                        # For counting time loop steps
eta_list_2 = list(); u_list_2 = list(); v_list_2 = list()         # Lists to contain eta and u,v for animation
hm_sample = list(); ts_sample = list(); t_sample = list()   # Lists for Hovmuller and time series
hm_sample.append(eta_n[:, int(N_y/2)])                      # Sample initial eta in middle of domain
ts_sample.append(eta_n[int(N_x/2), int(N_y/2)])             # Sample initial eta at center of domain
t_sample.append(0.0)                                        # Add initial time to t-samples
sample_interval = 1000                                      # How often to sample for time series
# =============== Done with setting up arrays and initial conditions ===============

t_0 = time.perf_counter()  # For timing the computation loop

# ==================================================================================
# ========================= Main time loop for simulation ==========================
# ==================================================================================
while (time_step < max_time_step):
    # ------------ Computing values for u and v at next time step --------------
    u_np1[:-1, :] = u_n[:-1, :] - g*dt/dx*(eta_n[1:, :] - eta_n[:-1, :])
    v_np1[:, :-1] = v_n[:, :-1] - g*dt/dy*(eta_n[:, 1:] - eta_n[:, :-1])

    # Add friction if enabled.
    if (use_friction is True):
        u_np1[:-1, :] -= dt*kappa[:-1, :]*u_n[:-1, :]
        v_np1[:-1, :] -= dt*kappa[:-1, :]*v_n[:-1, :]

    # Add wind stress if enabled.
    if (use_wind is True):
        u_np1[:-1, :] += dt*tau_x[:]/(rho_0*H)
        v_np1[:-1, :] += dt*tau_y[:]/(rho_0*H)

    # Use a corrector method to add coriolis if it's enabled.
    if (use_coriolis is True):
        u_np1[:, :] = (u_np1[:, :] - beta_c*u_n[:, :] + alpha*v_n[:, :])/(1 + beta_c)
        v_np1[:, :] = (v_np1[:, :] - beta_c*v_n[:, :] - alpha*u_n[:, :])/(1 + beta_c)
    
    v_np1[:, -1] = 0.0      # Northern boundary condition
    u_np1[-1, :] = 0.0      # Eastern boundary condition
    # -------------------------- Done with u and v -----------------------------

    # --- Computing arrays needed for the upwind scheme in the eta equation.----
    h_e[:-1, :] = np.where(u_np1[:-1, :] > 0, eta_n[:-1, :] + H, eta_n[1:, :] + H)
    h_e[-1, :] = eta_n[-1, :] + H

    h_w[0, :] = eta_n[0, :] + H
    h_w[1:, :] = np.where(u_np1[:-1, :] > 0, eta_n[:-1, :] + H, eta_n[1:, :] + H)

    h_n[:, :-1] = np.where(v_np1[:, :-1] > 0, eta_n[:, :-1] + H, eta_n[:, 1:] + H)
    h_n[:, -1] = eta_n[:, -1] + H

    h_s[:, 0] = eta_n[:, 0] + H
    h_s[:, 1:] = np.where(v_np1[:, :-1] > 0, eta_n[:, :-1] + H, eta_n[:, 1:] + H)

    uhwe[0, :] = u_np1[0, :]*h_e[0, :]
    uhwe[1:, :] = u_np1[1:, :]*h_e[1:, :] - u_np1[:-1, :]*h_w[1:, :]

    vhns[:, 0] = v_np1[:, 0]*h_n[:, 0]
    vhns[:, 1:] = v_np1[:, 1:]*h_n[:, 1:] - v_np1[:, :-1]*h_s[:, 1:]
    # ------------------------- Upwind computations done -------------------------

    # ----------------- Computing eta values at next time step -------------------
    eta_np1[:, :] = eta_n[:, :] - dt*(uhwe[:, :]/dx + vhns[:, :]/dy)    # Without source/sink

    # Add source term if enabled.
    if (use_source is True):
        eta_np1[:, :] += dt*sigma

    # Add sink term if enabled.
    if (use_sink is True):
        eta_np1[:, :] -= dt*w
    # ----------------------------- Done with eta --------------------------------

    u_n = np.copy(u_np1)        # Update u for next iteration
    v_n = np.copy(v_np1)        # Update v for next iteration
    eta_n = np.copy(eta_np1)    # Update eta for next iteration

    time_step += 1

    # Samples for Hovmuller diagram and spectrum every sample_interval time step.
    if (time_step % sample_interval == 0):
        hm_sample.append(eta_n[:, int(N_y/2)])              # Sample middle of domain for Hovmuller
        ts_sample.append(eta_n[int(N_x/2), int(N_y/2)])     # Sample center point for spectrum
        t_sample.append(time_step*dt)                       # Keep track of sample times.

    # Store eta and (u, v) every anin_interval time step for animations.
    if (time_step % anim_interval == 0):
        print("Time: \t{:.2f} hours".format(time_step*dt/3600))
        print("Step: \t{} / {}".format(time_step, max_time_step))
        print("Mass: \t{}\n".format(np.sum(eta_n)))
        u_list_2.append(u_n)
        v_list_2.append(v_n)
        eta_list_2.append(eta_n)
        
        rmse_temp = np.sqrt(np.mean((eta_n - eta_list[time_step // anim_interval - 1])**2))
        print("RMSE")
        print(rmse_temp)
        rmse_list_perturbed.append(rmse_temp)

# np.save("perturbed_rmse.npy", np.array(rmse_list_perturbed), allow_pickle = True)
# ======================EnKF=====================================================================================
ensemble_size = 200
rmse_list_cnn = []
# Useful linear algebra: compute B/A
import numpy.linalg as nla

xDim = N_x * N_y

# Useful linear algebra: compute B/A
import numpy.linalg as nla

xDim = N_x * N_y
latent_dim = 25*latent_channels
obs_sigma =  0.005
R12 = obs_sigma*np.eye((latent_dim))
R = R12 @ R12.T
def Obs(E):
    return E

def my_EnKF(state, obs, N, scaling):
    """My implementation of the EnKF."""
    ### Init ##
    E = state
       
    y = obs[:, None]  # current observation
    Eo = E            # observed ensemble
    # Compute ensemble moments
    Y = Eo - Eo.mean(keepdims=True)#, axis = 1)
    X = E - E.mean(keepdims=True)#, axis = 1)
    PH = X @ Y.T / (N-1)
    HPH = Y @ Y.T / (N-1)
    # Compute Kalman Gain
    KG = nla.solve(HPH + R, PH.T).T
    # Generate perturbations
    Perturbs = scaling * R12 @ rnd.randn(latent_dim, ensemble_size)
    # Update ensemble with KG
    E += KG @ (y - Eo - Perturbs)
    # Save statistics
    return E


# Define source array if source is enabled.
if (use_source):
    sigma = np.zeros((N_x, N_y))
    sigma = 0.0001*np.exp(-((X-L_x/2)**2/(2*(1E+5)**2) + (Y-L_y/2)**2/(2*(1E+5)**2)))
    
# Define source array if source is enabled.
if (use_sink is True):
    w = np.ones((N_x, N_y))*sigma.sum()/(N_x*N_y)

# Write all parameters out to file.
with open("param_output.txt", "w") as output_file:
    output_file.write(param_string)

print(param_string)     # Also print parameters to screen
u_n = np.zeros((ensemble_size, N_x, N_y))      # To hold u at current time step
u_np1 = np.zeros((ensemble_size, N_x, N_y))    # To hold u at next time step
v_n = np.zeros((ensemble_size, N_x, N_y))      # To hold v at current time step
v_np1 = np.zeros((ensemble_size, N_x, N_y))    # To hold v at enxt time step
eta_n = np.zeros((ensemble_size, N_x, N_y))    # To hold eta at current time step
eta_np1 = np.zeros((ensemble_size, N_x, N_y))  # To hold eta at next time step

# Temporary variables (each time step) for upwind scheme in eta equation
h_e = np.zeros((ensemble_size, N_x, N_y))
h_w = np.zeros((ensemble_size, N_x, N_y))
h_n = np.zeros((ensemble_size, N_x, N_y))
h_s = np.zeros((ensemble_size, N_x, N_y))
uhwe = np.zeros((ensemble_size, N_x, N_y))
vhns = np.zeros((ensemble_size, N_x, N_y))
state = torch.zeros(size = (ensemble_size, 19, 8, 5, 5)).to(device)

# Initial conditions for u and v.
u_n[:, :, :] = 0.0             # Initial condition for u
v_n[:, :, :] = 0.0             # Initial condition for u
u_n[:, -1, :] = 0.0            # Ensuring initial u satisfy BC
v_n[:, :, -1] = 0.0            # Ensuring initial v satisfy BC

# Initial condition for eta.
#eta_n[:, :] = np.sin(4*np.pi*X/L_y) + np.sin(4*np.pi*Y/L_y)
#eta_n = np.exp(-((X-0)**2/(2*(L_R)**2) + (Y-0)**2/(2*(L_R)**2)))
eta_n = np.exp(-((X-L_x/2)**2/(2*(0.05E+6)**2) + (Y-L_y/3)**2/(2*(0.05E+6)**2)))
eta_n = eta_n[None, :, :] + rnd.randn(ensemble_size, eta_n.shape[0], eta_n.shape[1])*0.05
#eta_n[int(3*N_x/8):int(5*N_x/8),int(3*N_y/8):int(5*N_y/8)] = 1.0
#eta_n[int(6*N_x/8):int(7*N_x/8),int(6*N_y/8):int(7*N_y/8)] = 1.0
#eta_n[int(3*N_x/8):int(5*N_x/8), int(13*N_y/14):] = 1.0
#eta_n[:, :] = 0.0

#viz_tools.surface_plot3D(X, Y, eta_n, (X.min(), X.max()), (Y.min(), Y.max()), (eta_n.min(), eta_n.max()))

# Sampling variables.
time_step = 1                        # For counting time loop steps
eta_list_enkf = list(); u_list_enkf = list(); v_list_enkf = list()         # Lists to contain eta and u,v for animation
hm_sample = list(); ts_sample = list(); t_sample = list()   # Lists for Hovmuller and time series
hm_sample.append(np.mean(eta_n, axis = 0)[:, int(N_y/2)])                      # Sample initial eta in middle of domain
ts_sample.append(np.mean(eta_n, axis = 0)[int(N_x/2), int(N_y/2)])             # Sample initial eta at center of domain
t_sample.append(0.0)                                        # Add initial time to t-samples
sample_interval = 1000                                      # How often to sample for time series
# =============== Done with setting up arrays and initial conditions ===============

t_0 = time.perf_counter()  # For timing the computation loop

# ==================================================================================
# ========================= Main time loop for simulation ==========================
# ==================================================================================
while (time_step < max_time_step):

    
    time_step += 1
    with torch.no_grad():
        if time_step <= 50:
            # ------------ Computing values for u and v at next time step --------------
            u_np1[:, :-1, :] = u_n[:, :-1, :] - g*dt/dx*(eta_n[:, 1:, :] - eta_n[:, :-1, :])
            v_np1[:, :, :-1] = v_n[:, :, :-1] - g*dt/dy*(eta_n[:, :, 1:] - eta_n[:, :, :-1])

            # Add friction if enabled.
            if (use_friction is True):
                u_np1[:, :-1, :] -= dt*kappa[:, :-1, :]*u_n[:, :-1, :]
                v_np1[:, :-1, :] -= dt*kappa[:, :-1, :]*v_n[:, :-1, :]

            # Add wind stress if enabled.
            if (use_wind is True):
                u_np1[:, :-1, :] += dt*tau_x[:, :]/(rho_0*H)
                v_np1[:, :-1, :] += dt*tau_y[:, :]/(rho_0*H)

            # Use a corrector method to add coriolis if it's enabled.
            if (use_coriolis is True):
                u_np1[:, :, :] = (u_np1[:, :, :] - beta_c*u_n[:, :, :] + alpha*v_n[:, :, :])/(1 + beta_c)
                v_np1[:, :, :] = (v_np1[:, :, :] - beta_c*v_n[:, :, :] - alpha*u_n[:, :, :])/(1 + beta_c)
            
            v_np1[:, :, -1] = 0.0      # Northern boundary condition
            u_np1[:, -1, :] = 0.0      # Eastern boundary condition
            # -------------------------- Done with u and v -----------------------------

            # --- Computing arrays needed for the upwind scheme in the eta equation.----
            h_e[:, :-1, :] = np.where(u_np1[:, :-1, :] > 0, eta_n[:, :-1, :] + H, eta_n[:, 1:, :] + H)
            h_e[:, -1, :] = eta_n[:, -1, :] + H

            h_w[:, 0, :] = eta_n[:, 0, :] + H
            h_w[:, 1:, :] = np.where(u_np1[:, :-1, :] > 0, eta_n[:, :-1, :] + H, eta_n[:, 1:, :] + H)

            h_n[:, :, :-1] = np.where(v_np1[:, :, :-1] > 0, eta_n[:, :, :-1] + H, eta_n[:, :, 1:] + H)
            h_n[:, :, -1] = eta_n[:, :, -1] + H

            h_s[:, :, 0] = eta_n[:, :, 0] + H
            h_s[:, :, 1:] = np.where(v_np1[:, :, :-1] > 0, eta_n[:, :, :-1] + H, eta_n[:, :, 1:] + H)

            uhwe[:, 0, :] = u_np1[:, 0, :]*h_e[:, 0, :]
            uhwe[:, 1:, :] = u_np1[:, 1:, :]*h_e[:, 1:, :] - u_np1[:, :-1, :]*h_w[:, 1:, :]

            vhns[:, :, 0] = v_np1[:, :, 0]*h_n[:, :, 0]
            vhns[:, :, 1:] = v_np1[:, :, 1:]*h_n[:, :, 1:] - v_np1[:, :, :-1]*h_s[:, :, 1:]
            # ------------------------- Upwind computations done -------------------------

            # ----------------- Computing eta values at next time step -------------------
            eta_np1[:, :, :] = eta_n[:, :, :] - dt*(uhwe[:, :, :]/dx + vhns[:, :, :]/dy)    # Without source/sink

            # Add source term if enabled.
            if (use_source is True):
                eta_np1[:, :, :] += dt*sigma

            # Add sink term if enabled.
            if (use_sink is True):
                eta_np1[:, :, :] -= dt*w
            # ----------------------------- Done with eta --------------------------------

            u_n = np.copy(u_np1)        # Update u for next iteration
            v_n = np.copy(v_np1)        # Update v for next iteration
            eta_n = np.copy(eta_np1)    # Update eta for next iteration
            state = torch.roll(state, shifts = -1, dims = 1)
            target = torch.Tensor(np.stack([u_n, v_n, eta_n], axis = 1)).to(device)
            mean_logvar = cnn_model.encode(target)
            state[:, -1, :, :, :] = mean_logvar
        else:
            state_predict = cnn_model.forward_latent(state)[:, -1, :, :, :]
            state = torch.roll(state, shifts = -1, dims = 1)
            state[:, -1, :, :, :] = state_predict
    if ((time_step + anim_interval//2) % anim_interval == 0):
        scaling = 5
        with torch.no_grad():
            target_true = torch.Tensor(np.stack([u_list[time_step // anim_interval - 1], v_list[time_step // anim_interval - 1], eta_list[time_step // anim_interval - 1]], axis = 0)).unsqueeze(0).to(device)
            mean_current, log_var_current = cnn_model.split(state[:, -1, :, :, :])
            # mean, logvar = cnn_model.split(cnn_model.encode(target_true))
            mean, logvar = cnn_model.split(cnn_model.encode_sample(target_true[:, :, ::15, ::15]))
            latent_assimilated = torch.Tensor(my_EnKF(mean_current.view(mean_current.shape[0], -1).detach().cpu().numpy().T, mean.detach().cpu().numpy().reshape(-1) + R12 @ rnd.randn(R12.shape[0]), ensemble_size, scaling)).to(device).T.reshape(ensemble_size, latent_channels, 5, 5)
            latent = cnn_model.decode(latent_assimilated).detach().cpu().numpy()
            eta_n = latent[:, 2 , :, :]
            u_n = latent[:, 0, :, :]
            v_n = latent[:, 1, :, :]
    # Samples for Hovmuller diagram and spectrum every sample_interval time step.
    if (time_step % sample_interval == 0):
        hm_sample.append(np.mean(eta_n, axis = 0)[:, int(N_y/2)])              # Sample middle of domain for Hovmuller
        ts_sample.append(np.mean(eta_n, axis = 0)[int(N_x/2), int(N_y/2)])     # Sample center point for spectrum
        t_sample.append(time_step*dt)                       # Keep track of sample times.

    # Store eta and (u, v) every anin_interval time step for animations.
    if (time_step % anim_interval == 0):
        print("Time")
        print(time.perf_counter() - t_0)
        t_0 = time.perf_counter()
        print("Time: \t{:.2f} hours".format(time_step*dt/3600))
        print("Step: \t{} / {}".format(time_step, max_time_step))
        print("Mass: \t{}\n".format(np.sum(np.mean(eta_n, axis = 0))))
        u_list_enkf.append(np.mean(u_n))
        v_list_enkf.append(np.mean(v_n))
        eta_list_enkf.append(np.mean(eta_n, axis = 0))
        rmse_temp = np.sqrt(np.mean((np.mean(eta_n, axis = 0) - eta_list[time_step // anim_interval - 1])**2))
        print("RMSE")
        print(rmse_temp)
        rmse_list_cnn.append(rmse_temp)

np.save("swe_dynamics_15_swe_enkf_sparse.npy", np.array(rmse_list_cnn), allow_pickle = True)


# Don't use numpy's mean, cov, but rather a `for` loop.
def estimate_mean_and_cov(E):
    xDim, N = E.shape

    x_bar = np.sum(E, axis=1)/N
    C_bar = np.zeros((xDim, xDim))
    for n in range(N):
        xc = (E[:, n] - x_bar)[:, None] # x_centered
        C_bar += xc @ xc.T
        # C_bar += np.outer(xc, xc)
    C_bar /= (N-1)

    return x_bar, C_bar



# ======================EnSF=====================================================================================
ensemble_size = 64
eps_alpha = 0.05
obs_sigma =  0.05
anim_interval = 20
def cond_alpha(t):
    # conditional information
    # alpha_t(0) = 1
    # alpha_t(1) = esp_alpha \approx 0
    return 1 - (1-eps_alpha)*t


def cond_sigma_sq(t):
    # conditional sigma^2
    # sigma2_t(0) = 0
    # sigma2_t(1) = 1
    # sigma(t) = t
    return t

# drift function of forward SDE
def f_func(t):
    # f=d_(log_alpha)/dt
    alpha_t = cond_alpha(t)
    f_t = -(1-eps_alpha) / alpha_t
    return f_t


def g_sq(t):
    # g = d(sigma_t^2)/dt -2f sigma_t^2
    d_sigma_sq_dt = 1
    g2 = d_sigma_sq_dt - 2*f_func(t)*cond_sigma_sq(t)
    return g2

def g_func(t):
    return np.sqrt(g_sq(t))
def g_tau(t):
    return 1-t

# generate sample with reverse SDE
def reverse_SDE(x0, score_likelihood=None, time_steps=100,
                drift_fun=f_func, diffuse_fun=g_func, alpha_fun=cond_alpha, sigma2_fun=cond_sigma_sq,  save_path=False):
    # x_T: sample from standard Gaussian
    # x_0: target distribution to sample from

    # reverse SDE sampling process
    # N1 = x_T.shape[0]
    # N2 = x0.shape[0]
    # d = x_T.shape[1]

    # Generate the time mesh
    dt = 1.0/time_steps

    # Initialization
    xt = torch.randn(ensemble_size, x0.shape[1], device=device)
    t = 1.0

    # define storage
    if save_path:
        path_all = [xt]
        t_vec = [t]

    # forward Euler sampling
    for i in range(time_steps):
        # prior score evaluation
        alpha_t = alpha_fun(t)
        sigma2_t = sigma2_fun(t)


        # Evaluate the diffusion term
        diffuse = diffuse_fun(t)

        # Evaluate the drift term
        # drift = drift_fun(t)*xt - diffuse**2 * score_eval

        # Update
        if score_likelihood is not None:
            xt += - dt*( drift_fun(t)*xt + diffuse**2 * ( (xt - alpha_t*x0)/sigma2_t) - diffuse**2 * score_likelihood(xt, t) ) \
                  + np.sqrt(dt)*diffuse*torch.randn_like(xt)
        else:
            xt += - dt*( drift_fun(t)*xt + diffuse**2 * ( (xt - alpha_t*x0)/sigma2_t) ) + np.sqrt(dt)*diffuse*torch.randn_like(xt)

        # Store the state in the path
        if save_path:
            path_all.append(xt)
            t_vec.append(t)

        # update time
        t = t - dt

    if save_path:
        return path_all, t_vec
    else:
        return xt
# Define source array if source is enabled.
if (use_source):
    sigma = np.zeros((N_x, N_y))
    sigma = 0.0001*np.exp(-((X-L_x/2)**2/(2*(1E+5)**2) + (Y-L_y/2)**2/(2*(1E+5)**2)))
    
# Define source array if source is enabled.
if (use_sink is True):
    w = np.ones((N_x, N_y))*sigma.sum()/(N_x*N_y)

# Write all parameters out to file.
with open("param_output.txt", "w") as output_file:
    output_file.write(param_string)

print(param_string)     # Also print parameters to screen
u_n = np.zeros((ensemble_size, N_x, N_y))      # To hold u at current time step
u_np1 = np.zeros((ensemble_size, N_x, N_y))    # To hold u at next time step
v_n = np.zeros((ensemble_size, N_x, N_y))      # To hold v at current time step
v_np1 = np.zeros((ensemble_size, N_x, N_y))    # To hold v at enxt time step
eta_n = np.zeros((ensemble_size, N_x, N_y))    # To hold eta at current time step
eta_np1 = np.zeros((ensemble_size, N_x, N_y))  # To hold eta at next time step

# Temporary variables (each time step) for upwind scheme in eta equation
h_e = np.zeros((ensemble_size, N_x, N_y))
h_w = np.zeros((ensemble_size, N_x, N_y))
h_n = np.zeros((ensemble_size, N_x, N_y))
h_s = np.zeros((ensemble_size, N_x, N_y))
uhwe = np.zeros((ensemble_size, N_x, N_y))
vhns = np.zeros((ensemble_size, N_x, N_y))
rmse_list_cnn_ensf = []
# Initial conditions for u and v.
u_n[:, :, :] = 0.0             # Initial condition for u
v_n[:, :, :] = 0.0             # Initial condition for u
u_n[:, -1, :] = 0.0            # Ensuring initial u satisfy BC
v_n[:, :, -1] = 0.0            # Ensuring initial v satisfy BC

# Initial condition for eta.
#eta_n[:, :] = np.sin(4*np.pi*X/L_y) + np.sin(4*np.pi*Y/L_y)
#eta_n = np.exp(-((X-0)**2/(2*(L_R)**2) + (Y-0)**2/(2*(L_R)**2)))
eta_n = np.exp(-((X-L_x/2)**2/(2*(0.05E+6)**2) + (Y-L_y/3)**2/(2*(0.05E+6)**2)))
eta_n = eta_n[None, :, :] + rnd.randn(ensemble_size, eta_n.shape[0], eta_n.shape[1])*0.05
#eta_n[int(3*N_x/8):int(5*N_x/8),int(3*N_y/8):int(5*N_y/8)] = 1.0
#eta_n[int(6*N_x/8):int(7*N_x/8),int(6*N_y/8):int(7*N_y/8)] = 1.0
#eta_n[int(3*N_x/8):int(5*N_x/8), int(13*N_y/14):] = 1.0
#eta_n[:, :] = 0.0

#viz_tools.surface_plot3D(X, Y, eta_n, (X.min(), X.max()), (Y.min(), Y.max()), (eta_n.min(), eta_n.max()))

# Sampling variables.
time_step = 1                        # For counting time loop steps
eta_list_ensf = list(); u_list_ensf = list(); v_list_ensf = list()         # Lists to contain eta and u,v for animation
hm_sample = list(); ts_sample = list(); t_sample = list()   # Lists for Hovmuller and time series
hm_sample.append(np.mean(eta_n, axis = 0)[:, int(N_y/2)])                      # Sample initial eta in middle of domain
ts_sample.append(np.mean(eta_n, axis = 0)[int(N_x/2), int(N_y/2)])             # Sample initial eta at center of domain
t_sample.append(0.0)                                        # Add initial time to t-samples
sample_interval = 1000                                      # How often to sample for time series
# =============== Done with setting up arrays and initial conditions ===============
scaling = 1
t_0 = time.perf_counter()  # For timing the computation loop

# ==================================================================================
# ========================= Main time loop for simulation ==========================
# ==================================================================================
while (time_step < 0):
    # ------------ Computing values for u and v at next time step --------------
    u_np1[:, :-1, :] = u_n[:, :-1, :] - g*dt/dx*(eta_n[:, 1:, :] - eta_n[:, :-1, :])
    v_np1[:, :, :-1] = v_n[:, :, :-1] - g*dt/dy*(eta_n[:, :, 1:] - eta_n[:, :, :-1])

    # Add friction if enabled.
    if (use_friction is True):
        u_np1[:, :-1, :] -= dt*kappa[:, :-1, :]*u_n[:, :-1, :]
        v_np1[:, :-1, :] -= dt*kappa[:, :-1, :]*v_n[:, :-1, :]

    # Add wind stress if enabled.
    if (use_wind is True):
        u_np1[:, :-1, :] += dt*tau_x[:, :]/(rho_0*H)
        v_np1[:, :-1, :] += dt*tau_y[:, :]/(rho_0*H)

    # Use a corrector method to add coriolis if it's enabled.
    if (use_coriolis is True):
        u_np1[:, :, :] = (u_np1[:, :, :] - beta_c*u_n[:, :, :] + alpha*v_n[:, :, :])/(1 + beta_c)
        v_np1[:, :, :] = (v_np1[:, :, :] - beta_c*v_n[:, :, :] - alpha*u_n[:, :, :])/(1 + beta_c)
    
    v_np1[:, :, -1] = 0.0      # Northern boundary condition
    u_np1[:, -1, :] = 0.0      # Eastern boundary condition
    # -------------------------- Done with u and v -----------------------------

    # --- Computing arrays needed for the upwind scheme in the eta equation.----
    h_e[:, :-1, :] = np.where(u_np1[:, :-1, :] > 0, eta_n[:, :-1, :] + H, eta_n[:, 1:, :] + H)
    h_e[:, -1, :] = eta_n[:, -1, :] + H

    h_w[:, 0, :] = eta_n[:, 0, :] + H
    h_w[:, 1:, :] = np.where(u_np1[:, :-1, :] > 0, eta_n[:, :-1, :] + H, eta_n[:, 1:, :] + H)

    h_n[:, :, :-1] = np.where(v_np1[:, :, :-1] > 0, eta_n[:, :, :-1] + H, eta_n[:, :, 1:] + H)
    h_n[:, :, -1] = eta_n[:, :, -1] + H

    h_s[:, :, 0] = eta_n[:, :, 0] + H
    h_s[:, :, 1:] = np.where(v_np1[:, :, :-1] > 0, eta_n[:, :, :-1] + H, eta_n[:, :, 1:] + H)

    uhwe[:, 0, :] = u_np1[:, 0, :]*h_e[:, 0, :]
    uhwe[:, 1:, :] = u_np1[:, 1:, :]*h_e[:, 1:, :] - u_np1[:, :-1, :]*h_w[:, 1:, :]

    vhns[:, :, 0] = v_np1[:, :, 0]*h_n[:, :, 0]
    vhns[:, :, 1:] = v_np1[:, :, 1:]*h_n[:, :, 1:] - v_np1[:, :, :-1]*h_s[:, :, 1:]
    # ------------------------- Upwind computations done -------------------------

    # ----------------- Computing eta values at next time step -------------------
    eta_np1[:, :, :] = eta_n[:, :, :] - dt*(uhwe[:, :, :]/dx + vhns[:, :, :]/dy)    # Without source/sink

    # Add source term if enabled.
    if (use_source is True):
        eta_np1[:, :, :] += dt*sigma

    # Add sink term if enabled.
    if (use_sink is True):
        eta_np1[:, :, :] -= dt*w
    # ----------------------------- Done with eta --------------------------------

    u_n = np.copy(u_np1)        # Update u for next iteration
    v_n = np.copy(v_np1)        # Update v for next iteration
    eta_n = np.copy(eta_np1)    # Update eta for next iteration
    time_step += 1
    euler_steps = 100
    # define likelihood score
    if (time_step % anim_interval == 0):
        # with torch.no_grad():
        #     target_true = torch.Tensor(np.stack([u_list[time_step // anim_interval - 1], v_list[time_step // anim_interval - 1], eta_list[time_step // anim_interval - 1]], axis = 0)).unsqueeze(0).to(device)
        #     target = torch.Tensor(np.stack([u_n, v_n, eta_n], axis = 1)).to(device)
        #     t_rep = torch.Tensor([time_step*dt/21/5000]*target.shape[0]).to(device)
        #     x_hat_current, mean_current, log_var_current = cnn_model((target, t_rep))
        #     x_hat, mean, log_var = cnn_model((target_true, t_rep))
        #     t_rep = torch.Tensor([time_step*dt]*target.shape[0]).to(device).unsqueeze(1)
        #     x_hat_sample, mean_sample, log_var_sample = cnn_model.forward_sample((target_true[:, :, ::30, ::30], t_rep))
        #     obs = torch.atan(mean.view(-1)*scaling)+ torch.randn_like(mean.view(-1))*obs_sigma
        # def score_likelihood(xt, t):
        #     # obs: (d)
        #     # xt: (ensemble, d)
        #     # the line below is analytical ∇z log P(Yt+1|z)
        #     score_x = -(torch.atan(xt) - obs)/obs_sigma**2 * (1./(1. + xt**2))
        #     tau = g_tau(t)
        #     return tau*score_x
        # mean_current = reverse_SDE(x0=mean_current.view(mean_current.shape[0], -1)*scaling, score_likelihood=score_likelihood, time_steps=euler_steps)
        # with torch.no_grad():
        #     latent = cnn_model.decode((mean_current.view(mean_current.shape[0], latent_channels, 5, 5)/scaling, t_rep)).detach().cpu().numpy()
        # eta_n = latent[:, 2, :, :]
        # u_n = latent[:, 0, :, :]
        # v_n = latent[:, 1, :, :]
        with torch.no_grad():
            target_true = torch.Tensor(np.stack([u_list[time_step // anim_interval - 1], v_list[time_step // anim_interval - 1], eta_list[time_step // anim_interval - 1]], axis = 0)).unsqueeze(0).to(device)
            target = torch.Tensor(np.stack([u_n, v_n, eta_n], axis = 1)).to(device)
            obs = torch.atan(target_true.view(-1)*scaling)+ torch.randn_like(target_true.view(-1))*obs_sigma
        def score_likelihood(xt, t):
            # obs: (d)
            # xt: (ensemble, d)
            # the line below is analytical ∇z log P(Yt+1|z)
            score_x = -(torch.atan(xt) - obs)/obs_sigma**2 * (1./(1. + xt**2))
            tau = g_tau(t)
            return tau*score_x
        mean_current = reverse_SDE(x0=target.view(target.shape[0], -1)*scaling, score_likelihood=score_likelihood, time_steps=euler_steps)
        with torch.no_grad():
            mean_current = (mean_current.view(mean_current.shape[0], 3, 150, 150)/scaling).detach().cpu().numpy()
        eta_n = mean_current[:, 2, :, :]
        u_n = mean_current[:, 0, :, :]
        v_n = mean_current[:, 1, :, :]
    # Samples for Hovmuller diagram and spectrum every sample_interval time step.
    if (time_step % sample_interval == 0):
        hm_sample.append(np.mean(eta_n, axis = 0)[:, int(N_y/2)])              # Sample middle of domain for Hovmuller
        ts_sample.append(np.mean(eta_n, axis = 0)[int(N_x/2), int(N_y/2)])     # Sample center point for spectrum
        t_sample.append(time_step*dt)                       # Keep track of sample times.

    # Store eta and (u, v) every anin_interval time step for animations.
    if (time_step % anim_interval == 0):
        print("Time: \t{:.2f} hours".format(time_step*dt/3600))
        print("Step: \t{} / {}".format(time_step, max_time_step))
        print("Mass: \t{}\n".format(np.sum(np.mean(eta_n, axis = 0))))
        u_list_ensf.append(np.mean(u_n, axis = 0))
        v_list_ensf.append(np.mean(v_n, axis = 0))
        eta_list_ensf.append(np.mean(eta_n, axis = 0))
        rmse_temp = np.sqrt(np.mean((np.mean(eta_n, axis = 0) - eta_list[time_step // anim_interval - 1])**2))
        print("RMSE")
        print(rmse_temp)
        rmse_list_cnn_ensf.append(rmse_temp)

np.save("cnn_full_ensf_RMSE.npy", np.array(rmse_list_cnn), allow_pickle = True)


# ============================= Main time loop done ================================
print("Main computation loop done!\nExecution time: {:.2f} s".format(time.perf_counter() - t_0))
print("\nVisualizing results...")

# ==================================================================================
# ================== Visualizing results by call to external file ==================
# ==================================================================================
#viz_tools.pmesh_plot(X, Y, eta_n, "Final state of surface elevation $\eta$")
#viz_tools.quiver_plot(X, Y, u_n, v_n, "Final state of velocity field $\mathbf{u}(x,y)$")
#viz_tools.hovmuller_plot(x, t_sample, hm_sample)
#viz_tools.plot_time_series_and_ft(t_sample, ts_sample)
eta_anim = viz_tools.eta_animation_overlay(X, Y, eta_list, eta_list_2, eta_list_enkf, eta_list, ["Perturbed", "EnKF", "$\eta$"], anim_interval*dt, "eta")
#eta_surf_anim = viz_tools.eta_animation3D(X, Y, eta_list, anim_interval*dt, "eta_surface")
#quiv_anim = viz_tools.velocity_animation(X, Y, u_list, v_list, anim_interval*dt, "velocity")
# ============================ Done with visualization =============================

print("\nVisualization done!")
plt.show()
