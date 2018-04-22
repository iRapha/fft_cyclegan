import PIL
from torchvision import transforms
import torch
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import torch.nn.functional as F

from torch.autograd import Variable 
Var = lambda x : Variable(x, requires_grad=False)

SOURCE = '1005_A_fake_B.png'
MASK = '1005_A_fake_B_mask.png'


def get_kernel(size, sigma):
    temp = np.zeros((size, size))
    temp[size//2, size//2] = 1
    return gaussian_filter(temp, sigma).copy()

def kernel(size, sigma):
    k = torch.FloatTensor(get_kernel(size, sigma))
    #handle each channel filter
    r_f = torch.zeros(3, size, size)
    r_f[0] = k.clone()

    g_f = torch.zeros(3, size, size)
    g_f[1] = k.clone()

    b_f = torch.zeros(3, size, size)
    b_f[2] = k.clone()
    return torch.stack([r_f, g_f, b_f])

def smooth(source, k, sigma = 1):
    return F.conv2d(Var(source.view(1, 3, 256, 256)),
                    Var(kernel(k, sigma)), padding=k//2)[0].data

# apply content to the areas specified in the mask
# mask can have float values
def interpolate_content(source, mask, content):
    return (source * (1 - mask) + content * (mask))

source = transforms.ToTensor()(PIL.Image.open(SOURCE))
#remove alpha channel from mask
mask = transforms.ToTensor()(PIL.Image.open(MASK))[:-1]

#get a smoothed version
content = smooth(source, 5, 1)

#apply the filter
t = interpolate_content(source, mask, content)
