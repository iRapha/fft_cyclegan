import torch
import pytorch_fft.fft as fft
import numpy as np

from PIL import Image

imreal = torch.from_numpy(np.array(Image.open('rome.png').convert('LA'))).cuda()
Image.fromarray(imreal.cpu().numpy(), mode='LA').save('preimg.png')
imreal = imreal.unsqueeze(0).float()
imimag = torch.zeros(*imreal.size()).cuda()


fft_version = fft.fft2(imreal, imimag)[0]
newimreal, newimimag = fft.ifft2(fft_version, torch.zeros(*fft_version.size()).cuda())

def pre(im):
    #  im = im.abs().log()
    #  mini = torch.min(im)
    #  maxi = torch.max(im)
    #  im = 255 * (im - mini) / (maxi - mini)
    #  print(im.min())
    #  print(im.max())
    return im.byte()

real_img = pre(newimreal).cpu().numpy()
imag_img = pre(newimimag).cpu().numpy()

#  np.save('real_img.npy', real_img)
Image.fromarray(real_img.squeeze(), mode='LA').save('real_img.png')
#  Image.fromarray(imag_img.squeeze(), mode='LA').save('real_img.png')

#  def norma(img):
    #  return img.squeeze()
    #  #  return 20*np.log(np.abs(np.fft.fftshift(img.squeeze())))
    #  #  img_min = np.min(img)
    #  #  img_max = np.max(img)
    #  #  return 255 * (img.squeeze() - img_min) / (img_max - img_min)

#  print(np.shape(breal_img))
#  print(np.shape(bimag_img))

#  Image.fromarray(norma(breal_img), mode='L').save('breal.png')
#  Image.fromarray(norma(bimag_img), mode='L').save('bimag.png')
