import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# Define the Generator Network
class Generator(nn.Module):
    def __init__(self, params):
        super().__init__()

        # Input is the latent vector Z.
        self.tconv1 = nn.ConvTranspose2d(params['nz'], params['ngf']*16,
            kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(params['ngf']*16)

        # Input Dimension: (ngf*8) x 4 x 4
        self.tconv2 = nn.ConvTranspose2d(params['ngf']*16, params['ngf']*8,
            8, 4, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(params['ngf']*8)

        # Input Dimension: (ngf*4) x 8 x 8
        self.tconv3 = nn.ConvTranspose2d(params['ngf']*8, params['ngf']*4,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(params['ngf']*4)

        # Input Dimension: (ngf*2) x 16 x 16
        self.tconv4 = nn.ConvTranspose2d(params['ngf']*4, params['ngf']*2,
            8, 4, 2, bias=False)
        self.bn4 = nn.BatchNorm2d(params['ngf']*2)

        # Input Dimension: (ngf*2) x 16 x 16
        self.tconv5 = nn.ConvTranspose2d(params['ngf']*2, params['ngf'],
            4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(params['ngf'])

        # Input Dimension: (ngf) * 32 * 32
        self.tconv6 = nn.ConvTranspose2d(params['ngf'], params['nc'],
            8, 4, 2, bias=False)
        #Output Dimension: (nc) x 64 x 64

    def forward(self, x):
        #print(x.shape, " v")
        x = F.relu(self.bn1(self.tconv1(x)))
        #print(x.shape, " w")
        x = F.relu(self.bn2(self.tconv2(x)))
        #print(x.shape, " x")
        x = F.relu(self.bn3(self.tconv3(x)))
        #print(x.shape, " y")
        x = F.relu(self.bn4(self.tconv4(x)))
        #print(x.shape, " z")
        x = F.relu(self.bn5(self.tconv5(x)))

        x = F.tanh(self.tconv6(x))

        return x

# Define the Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, params):
        super().__init__()

        # Input Dimension: (nc) x 64 x 64
        self.conv1 = nn.Conv2d(params['nc'], params['ndf'],
            8, 4, 2, bias=False)

        # Input Dimension: (ndf) x 32 x 32
        self.conv2 = nn.Conv2d(params['ndf'], params['ndf']*2,
            4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(params['ndf']*2)

        # Input Dimension: (ndf*2) x 16 x 16
        self.conv3 = nn.Conv2d(params['ndf']*2, params['ndf']*4,
            8, 4, 2, bias=False)
        self.bn3 = nn.BatchNorm2d(params['ndf']*4)

        # Input Dimension: (ndf*4) x 8 x 8
        self.conv4 = nn.Conv2d(params['ndf']*4, params['ndf']*8,
            4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(params['ndf']*8)
        
        # Input Dimension: (ndf*4) x 8 x 8
        self.conv5 = nn.Conv2d(params['ndf']*8, params['ndf']*16,
            8, 4, 2, bias=False)
        self.bn5 = nn.BatchNorm2d(params['ndf']*16)

        # Input Dimension: (ndf*8) x 4 x 4
        self.conv6 = nn.Conv2d(params['ndf']*16, 1, 4, 2, 1, bias=False)

        self.flatten = nn.Flatten()

    def forward(self, x):
        #print(x.shape, " a")
        x = F.leaky_relu(self.conv1(x), 0.2, True)
        #print(x.shape, " b")
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, True)
        #print(x.shape, " c")
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, True)
        #print(x.shape, " d")
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2, True)
        #print(x.shape, " e")
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.2, True)
        #print(x.shape, " f")

        x = F.sigmoid(self.conv6(x))
        #print(x.shape, " g")

        return x

#parser = argparse.ArgumentParser()
#parser.add_argument('-load_path', default='model/model_final.pth', help='Checkpoint to load path from')
#parser.add_argument('-num_output', default=64, help='Number of generated outputs')
#args = parser.parse_args()

load_path = "model.pth"
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
noise = torch.randn(int(num_output), params['nz'], 1, 1, device=device)

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