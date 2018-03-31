import numpy as np
import torch
import pytorch_fft.fft as fft

from PIL import Image

im = np.array(Image.open('rome.jpg').convert('LA'), dtype=np.uint8)
#  Image.fromarray(im, mode='LA').save('newrome.png')

print('min: {}, max: {}'.format(np.min(im), np.max(im)))

im = torch.from_numpy(im).float().cuda()
im = im.permute(2, 0, 1).contiguous()
print(im.type())
print(im.size())

B_real, B_imag = fft.fft2(im, torch.zeros(*im.size()).cuda())
im, _ = fft.ifft2(B_real, torch.zeros(*im.size()).cuda())

im = im.permute(1, 2, 0).contiguous()
im = im.byte().cpu().numpy()
Image.fromarray(im, mode='LA').save('newrometorched.png')

print('min: {}, max: {}'.format(np.min(im), np.max(im)))
