import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.utils as vutils
import numpy as np
import random
import torch.optim as optim

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

root = 'kikuwu_img'

def get_dataloader(params):
    """
    Loads the dataset and applies proproccesing steps to it.
    Returns a PyTorch DataLoader.

    """
    # Data proprecessing.
    transform = transforms.Compose([
        transforms.Resize(params['imsize']),
        transforms.CenterCrop(params['imsize']),
        transforms.Grayscale(),
        Phaseshift(),
        transforms.ToTensor(),
        Flatten()])

    # Create the dataset.
    dataset = dset.ImageFolder(root=root, transform=transform)
    print(dataset)

    # Create the dataloader.
    dataloader = torch.utils.data.DataLoader(dataset,
        batch_size=params['bsize'],
        shuffle=True)

    return dataloader

class Phaseshift(object):
    def __init__(self):
        pass
    
    def __call__(self, image):
        dimension = 512
        n = random.randint(-dimension,dimension)
        image = np.concatenate((image, image, image), axis=0)[dimension+n:2*dimension+n]

        return image#{'image': image, 'landmarks': landmarks}
    
class Flatten(object):
    def __init__(self):
        pass
    
    def __call__(self, image):
        image = image.view(1, -1)

        return image#{'image': image, 'landmarks': landmarks}

# Generator Network
#try: batchnorm?
class Generator(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.tconv1 = nn.ConvTranspose1d(100, 1024,
            kernel_size=16, stride=8, padding=4, bias=False)

        self.tconv2 = nn.ConvTranspose1d(1024, 512,
            16, 8, 4, bias=False)

        self.tconv3 = nn.ConvTranspose1d(512, 256,
            16, 8, 4, bias=False)

        self.tconv4 = nn.ConvTranspose1d(256, 128,
            16, 8, 4, bias=False)

        self.tconv5 = nn.ConvTranspose1d(128, 64,
            16, 8, 4, bias=False)

        self.tconv6 = nn.ConvTranspose1d(64, 1,
            16, 8, 4, bias=False)

    def forward(self, x):
        debug = 0
        if debug == 1: print(x.shape, " t")
        x = F.relu(self.tconv1(x))
        if debug == 1: print(x.shape, " u")
        x = F.relu(self.tconv2(x))
        if debug == 1: print(x.shape, " v")
        x = F.relu(self.tconv3(x))
        if debug == 1: print(x.shape, " w")
        x = F.relu(self.tconv4(x))
        if debug == 1: print(x.shape, " x")
        x = F.relu(self.tconv5(x))
        if debug == 1: print(x.shape, " y")
        x = F.tanh(self.tconv6(x))
        if debug == 1: print(x.shape, " z")
        return x

# Define the Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 64,
            1024, 512, 256, bias=False)

        # Input Dimension: (ndf) x 32 x 32
        self.conv2 = nn.Conv1d(64, 1,
            1024, 512, 256, bias=False)

    def forward(self, x):
        debug = 1
        if debug == 1: print(x.shape, " a")
        x = F.leaky_relu(self.conv1(x), 0.2, True)
        if debug == 1: print(x.shape, " b")
        x = F.leaky_relu(self.conv2(x), 0.2, True)

        x = F.sigmoid(x)
        if debug == 1: print(x.shape, " c")

        return x
    
# Set random seed for reproducibility.
seed = 369
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

# Use GPU is available else use CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
print(device, " will be used.\n")

dataloader = get_dataloader(params)

def weights_init(w):
    """
    Initializes the weights of the layer, w.
    """
    classname = w.__class__.__name__
    if classname.find('conv') != -1:
        nn.init.normal_(w.weight.data, 0.0, 0.02)
    elif classname.find('bn') != -1:
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)

# Create the generator.
netG = Generator(params).to(device)
# Apply the weights_init() function to randomly initialize all
# weights to mean=0.0, stddev=0.2
netG.apply(weights_init)
# Print the model.
print(netG)

# Create the discriminator.
netD = Discriminator(params).to(device)
# Apply the weights_init() function to randomly initialize all
# weights to mean=0.0, stddev=0.2
netD.apply(weights_init)
# Print the model.
print(netD)


# Binary Cross Entropy loss function.
criterion = nn.BCELoss()

fixed_noise = torch.randn(64, params['nz'], 1, device=device)

real_label = 1
fake_label = 0

# Optimizer for the discriminator.
optimizerD = optim.Adam(netD.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))
# Optimizer for the generator.
optimizerG = optim.Adam(netG.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))

# Stores generated images as training progresses.
img_list = []
# Stores generator losses during training.
G_losses = []
# Stores discriminator losses during training.
D_losses = []

iters = 0

print("Starting Training Loop...")
print("-"*25)

for epoch in range(params['nepochs']):
    print(dataloader.dataset)
    for i, data in enumerate(dataloader, 0):
        # Transfer data tensor to GPU/CPU (device)
        real_data = data[0].to(device)
        # Get batch size. Can be different from params['nbsize'] for last batch in epoch.
        b_size = real_data.size(0)
        
        # Make accumalated gradients of the discriminator zero.
        netD.zero_grad()
        # Create labels for the real data. (label=1)
        label = torch.full((b_size, ), real_label, device=device)
        #print(real_data.shape)
        output = netD(real_data).view(-1)
        #print(output.shape)
        #print(label.shape)
        errD_real = criterion(output, label.float())
        # Calculate gradients for backpropagation.
        errD_real.backward()
        D_x = output.mean().item()
        
        # Sample random data from a unit normal distribution.
        noise = torch.randn(b_size, params['nz'], 1, device=device)
        # Generate fake data (images).
        fake_data = netG(noise)
        # Create labels for fake data. (label=0)
        label.fill_(fake_label)
        # Calculate the output of the discriminator of the fake data.
        # As no gradients w.r.t. the generator parameters are to be
        # calculated, detach() is used. Hence, only gradients w.r.t. the
        # discriminator parameters will be calculated.
        # This is done because the loss functions for the discriminator
        # and the generator are slightly different.
        output = netD(fake_data.detach()).view(-1)

        errD_fake = criterion(output, label.float())
        # Calculate gradients for backpropagation.
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        # Net discriminator loss.
        errD = errD_real + errD_fake
        # Update discriminator parameters.
        optimizerD.step()
        
        # Make accumalted gradients of the generator zero.
        netG.zero_grad()
        # We want the fake data to be classified as real. Hence
        # real_label are used. (label=1)
        label.fill_(real_label)
        # No detach() is used here as we want to calculate the gradients w.r.t.
        # the generator this time.
        output = netD(fake_data).view(-1)
        errG = criterion(output, label.float())
        # Gradients for backpropagation are calculated.
        # Gradients w.r.t. both the generator and the discriminator
        # parameters are calculated, however, the generator's optimizer
        # will only update the parameters of the generator. The discriminator
        # gradients will be set to zero in the next iteration by netD.zero_grad()
        errG.backward()

        D_G_z2 = output.mean().item()
        # Update generator parameters.
        optimizerG.step()

        # Check progress of training.
        if i % 1 == 0:
            print(torch.cuda.is_available())
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, params['nepochs'], i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save the losses for plotting.
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on a fixed noise.
        if (iters % 100 == 0) or ((epoch == params['nepochs']-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake_data = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake_data, padding=2, normalize=True))

        iters += 1

    # Save the model.
    if epoch % params['save_epoch'] == 0:
        torch.save({
            'generator' : netG.state_dict(),
            'discriminator' : netD.state_dict(),
            'optimizerG' : optimizerG.state_dict(),
            'optimizerD' : optimizerD.state_dict(),
            'params' : params
            }, 'model_{}.pth'.format(epoch))