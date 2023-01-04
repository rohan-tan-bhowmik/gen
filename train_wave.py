import torch
import torch.nn as nn
import torch.nn.functional as F

# Parameters to define the model.
params = {
    "bsize" : 128,# Batch size during training.
    'imsize' : 512,# Spatial size of training images. All images will be resized to this size during preprocessing.
    'nc' : 1,# Number of channles in the training images. For coloured images this is 3.
    'nz' : 100,# Size of the Z latent vector (the input to the generator).
    'ngf' : 64,# Size of feature maps in the generator. The depth will be multiples of this.
    'ndf' : 64, # Size of features maps in the discriminator. The depth will be multiples of this.
    'nepochs' : 200,# Number of training epochs.
    'lr' : 0.00025,# Learning rate for optimizers
    'beta1' : 0.5,# Beta1 hyperparam for Adam optimizer
    'save_epoch' : 10}# Save step.

# Define the Generator Network
#try: batchnorm?
class Generator(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.tconv1 = nn.ConvTranspose2d(100, 1024*16,
            kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(params['ngf']*16)

        self.tconv2 = nn.ConvTranspose2d(1024*16, 512*64,
            8, 4, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(params['ngf']*8)

        self.tconv3 = nn.ConvTranspose2d(512*64, 256*256,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(params['ngf']*4)

        self.tconv4 = nn.ConvTranspose2d(256*256, 128*1024,
            8, 4, 2, bias=False)
        self.bn4 = nn.BatchNorm2d(params['ngf']*2)

        self.tconv5 = nn.ConvTranspose2d(128*1024, 64*4096,
            4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(params['ngf'])

        self.tconv6 = nn.ConvTranspose2d(64*4096, params['nc'],
            8, 4, 2, bias=False)
        
        #self.tconv7 = nn.ConvTranspose2d(64*4096, params['nc'],
        #    8, 4, 2, bias=False)

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