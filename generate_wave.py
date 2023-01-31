import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# Generator Network
#try: batchnorm?
class Generator(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.tconv1 = nn.ConvTranspose1d(100, 100,
            kernel_size=1024, stride=64, padding=480, bias=False)

        self.tconv2 = nn.ConvTranspose1d(100, 10,
            kernel_size=1024, stride=64, padding=480, bias=False)

        self.tconv3 = nn.ConvTranspose1d(10, 1,
            kernel_size=1024, stride=64, padding=480, bias=False)

    def forward(self, x):
        debug = 0
        if debug == 1: print(x.shape, " w")
        x = F.relu(self.tconv1(x))
        if debug == 1: print(x.shape, " x")
        x = F.relu(self.tconv2(x))
        if debug == 1: print(x.shape, " y")
        x = F.tanh(self.tconv3(x))
        if debug == 1: print(x.shape, " z")
        return x

# Define the Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 10,
            kernel_size=1024, stride=64, padding=480, bias=False)

        # Input Dimension: (ndf) x 32 x 32
        self.conv2 = nn.Conv1d(10, 100,
            kernel_size=1024, stride=64, padding=480, bias=False)
        
        # Input Dimension: (ndf) x 32 x 32
        self.conv3 = nn.Conv1d(100, 1,
            kernel_size=1024, stride=64, padding=480, bias=False)

    def forward(self, x):
        debug = 0
        if debug == 1: print(x.shape, " a")
        x = F.leaky_relu(self.conv1(x), 0.2, True)
        if debug == 1: print(x.shape, " b")
        x = F.leaky_relu(self.conv2(x), 0.2, True)
        if debug == 1: print(x.shape, " c")
        x = F.leaky_relu(self.conv3(x), 0.2, True)

        x = F.sigmoid(x)
        if debug == 1: print(x.shape, " d")

        return x

#parser = argparse.ArgumentParser()
#parser.add_argument('-load_path', default='model/model_final.pth', help='Checkpoint to load path from')
#parser.add_argument('-num_output', default=64, help='Number of generated outputs')
#args = parser.parse_args()

load_path = "model_4.pth"
num_output = 16

# Load the checkpoint file.
state_dict = torch.load(load_path)

# Set the device to run on: GPU or CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
# Get the 'params' dictionary from the loaded state_dict.
params = state_dict['params']

# Create the generator network.
netG = Generator(params).to(device)
# Load the trained generator weights.
netG.load_state_dict(state_dict['generator'])
print(netG)

print(num_output)
# Get latent vector Z from unit normal distribution.
noise = torch.randn(int(num_output), params['nz'], 1, device=device)

# Turn off gradient calculation to speed up the process.
with torch.no_grad():
	# Get generated image from the noise vector using
	# the trained generator.
    generated_img = netG(noise).detach().cpu()

# Display the generated image.
plt.axis("off")
plt.title("Generated Images")
plt.imshow(np.transpose(vutils.make_grid(generated_img, nrow = int(num_output**0.5), padding=2, normalize=False), (1,2,0)))

for i in range(num_output):
	plt.imsave("sample_{}.png".format(i), generated_img[i][0], cmap='gray')

plt.show()