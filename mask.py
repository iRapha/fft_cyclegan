import PIL
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from scipy.ndimage.filters import gaussian_filter
from torchvision import transforms
from torch.autograd import Variable
Var = lambda x : Variable(x, requires_grad=False)

SOURCE = 'mask_vis/1005_A_fake_B.png'
MASK = 'mask_vis/1005_A_fake_B_mask.png'
OUT = 'mask_vis/1005_A_fake_B_smoothed.png'


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


def get_fill(img, mask):
    mask = (mask>0).float() * 1.0
    r = (mask*img)[0]
    g = (mask*img)[1]
    b = (mask*img)[2]
    r = r.view(-1)
    g = g.view(-1)
    b = b.view(-1)
    r_nozeros = torch.Tensor([x for x in r if x > 0])
    g_nozeros = torch.Tensor([x for x in g if x > 0])
    b_nozeros = torch.Tensor([x for x in b if x > 0])
    print(len(r_nozeros))
    print('mean_r = {}'.format(torch.mean(r_nozeros)))
    print('mean_g = {}'.format(torch.mean(g_nozeros)))
    print('mean_b = {}'.format(torch.mean(b_nozeros)))
    r = torch.FloatTensor(256, 256).fill_(torch.mean(r_nozeros))
    g = torch.FloatTensor(256, 256).fill_(torch.mean(g_nozeros))
    b = torch.FloatTensor(256, 256).fill_(torch.mean(b_nozeros))
    return torch.stack([r, g, b])
    #  non_zero = [x for x in img * mask if x > 0]

def tocv2imshow(img):
    return np.array(transforms.ToPILImage()(img))


source = transforms.ToTensor()(PIL.Image.open(SOURCE).convert('RGB'))
#remove alpha channel from mask
mask = transforms.ToTensor()(PIL.Image.open(MASK))[:-1]

#get a smoothed version
#  content = smooth(source, 5, 1)
content = get_fill(source, mask)

#  cv2.imshow('content', tocv2imshow(content))
#  cv2.imshow('src', tocv2imshow(source - (source*mask)))
#  cv2.waitKey(0)
#  cv2.destroyAllWindows()

#apply the filter
t = interpolate_content(source, mask, content)

print(torch.mean(source))
print(torch.mean(t))

np_t = np.array(transforms.ToPILImage()(t))
cv2.imwrite(OUT, cv2.cvtColor(np_t, cv2.COLOR_BGR2RGB))
