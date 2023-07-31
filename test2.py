import matplotlib.pyplot as plt
import glob
import torch

sums = torch.zeros(2049)
for file in glob.iglob("kikuwu_specs/*.png"):
    img = plt.imread(file)[:,:,0]
    sums += torch.sum(torch.from_numpy(img)**2, dim=-1)
    print(file)

plt.plot(sums**0.5)
plt.plot(torch.zeros(2049))
plt.show()