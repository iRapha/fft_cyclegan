import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from PIL import Image
from skimage.color import rgb2lab, lab2rgb


def to255(img, numpy=True):
    if numpy:
        min_i = np.min(img)
        max_i = np.max(img)
    else:
        min_i = torch.min(img)
        max_i = torch.max(img)
    diff_i = max_i - min_i
    if diff_i == 0:
        diff_i = 1
    return (255.0 * (img - min_i) / diff_i)

def to100(img, numpy=True):
    if numpy:
        min_i = np.min(img)
        max_i = np.max(img)
    else:
        min_i = torch.min(img)
        max_i = torch.max(img)
    diff_i = max_i - min_i
    if diff_i == 0:
        diff_i = 1
    return (100.0 * (img - min_i) / diff_i)

def get_transform(norm_255=False, just_norm=False, png=False):
    transform_list = []

    # resize and crop:
    #  osize = [opt.loadSize, opt.loadSize] 286
    #  transform_list.append(transforms.Scale(osize, Image.BICUBIC))
    #  transform_list.append(transforms.RandomCrop(opt.fineSize)) 256

    if not just_norm:
        if png:
            osize = [256, 256]
        else:
            osize = [286, 286]
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fineSize))

        #  if opt.isTrain and not opt.no_flip:
            #  transform_list.append(transforms.RandomHorizontalFlip())
    if norm_255:
        transform_list += [transforms.ToTensor(),
                           transforms.Lambda(lambda x: to255(x, numpy=False)),
                           transforms.ToPILImage()]
    else:
        transform_list += [transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5)),
                           transforms.ToPILImage()]
    return transforms.Compose(transform_list)

def toLab(img):
    return rgb2lab(img)

def fromLab(img):
    return lab2rgb(img)


def get_diff(img1, img2, strategy='to255'):
    if strategy == 'naive':
        trans = get_transform(norm_255=True, just_norm=True)
        return np.array(trans(img1)) - np.array(trans(img2))
    elif strategy == 'to255':
        return to255(np.array(to255(img1)) - np.array(to255(img2)))
    elif strategy == 'wholelab':
        img1_lab = toLab(img1)
        img2_lab = toLab(img2)
        diff = np.abs(img1_lab - img2_lab)
        return (np.array(fromLab(diff)))
    elif strategy == 'wholelabto255':
        img1_lab = toLab(img1)
        img2_lab = toLab(img2)
        diff = np.abs(img1_lab - img2_lab)
        return np.array(to255(fromLab(diff)))
    elif strategy == 'justLzeroAB':
        img1_lab = np.expand_dims(toLab(img1)[:,:,0], -1)
        img2_lab = np.expand_dims(toLab(img2)[:,:,0], -1)
        diff = np.abs(img1_lab - img2_lab)
        zero_c = np.zeros((256, 256, 1))
        return np.array(to255(fromLab(np.concatenate((diff, zero_c, zero_c), axis=2))))
    elif strategy == 'justLvislab':
        img1_lab = np.expand_dims(toLab(img1)[:,:,0], -1)
        img2_lab = np.expand_dims(toLab(img2)[:,:,0], -1)
        diff = np.abs(img1_lab - img2_lab)
        return np.array(to255(np.concatenate((diff, diff, diff), axis=2)))
    elif strategy == 'justL1stAB':
        img1_lab = np.expand_dims(toLab(img1)[:,:,0], -1)
        img2_lab = np.expand_dims(toLab(img2)[:,:,0], -1)
        img1_lab_a = np.expand_dims(toLab(img1)[:,:,1], -1)
        img1_lab_b = np.expand_dims(toLab(img1)[:,:,2], -1)
        diff = np.abs(img1_lab - img2_lab)
        return np.array(to255(fromLab(np.concatenate((diff, img1_lab_a, img1_lab_b), axis=2))))
    elif strategy == 'justL2ndAB':
        img1_lab = np.expand_dims(toLab(img1)[:,:,0], -1)
        img2_lab = np.expand_dims(toLab(img2)[:,:,0], -1)
        img2_lab_a = np.expand_dims(toLab(img2)[:,:,1], -1)
        img2_lab_b = np.expand_dims(toLab(img2)[:,:,2], -1)
        diff = np.abs(img1_lab - img2_lab)
        return np.array(to255(fromLab(np.concatenate((diff, img2_lab_a, img2_lab_b), axis=2))))
    elif strategy == 'justLaverageAB':
        img1_lab = np.expand_dims(toLab(img1)[:,:,0], -1)
        img2_lab = np.expand_dims(toLab(img2)[:,:,0], -1)
        img1_lab_a = np.expand_dims(toLab(img1)[:,:,1], -1)
        img1_lab_b = np.expand_dims(toLab(img1)[:,:,2], -1)
        img2_lab_a = np.expand_dims(toLab(img2)[:,:,1], -1)
        img2_lab_b = np.expand_dims(toLab(img2)[:,:,2], -1)
        a = (img1_lab_a + img2_lab_a) / 2.0
        b = (img1_lab_b + img2_lab_b) / 2.0
        diff = np.abs(img1_lab - img2_lab)
        return np.array(to255(fromLab(np.concatenate((diff, a, b), axis=2))))

if __name__ == '__main__':
    fakeB_smoothest = np.array(Image.open('diff_vis/fakeBto_fake_A.png').convert('RGB'))
    fakeB = np.array(Image.open('diff_vis/1005_A_rec_A.png').convert('RGB'))

    #  cv2.imwrite('diff_vis/1005_A_rec_A_smoothest_normd.png', to255(fakeB_smoothest))
    #  cv2.imwrite('diff_vis/1005_A_rec_A_normd.png', to255(fakeB))


    #  trans = get_transform(norm_255=True, just_norm=True)
    #  cv2.imwrite('diff_vis/1005_A_rec_A_smoothest_normd.png', np.array(trans(fakeB_smoothest)))
    #  cv2.imwrite('diff_vis/1005_A_rec_A_normd.png', np.array(trans(fakeB)))

    #  print(fakeB)
    #  print(np.array(trans(fakeB)))

    cv2.imwrite('diff_vis/1005_A_rec_A_diff.png', get_diff(fakeB, fakeB_smoothest, strategy='to255'))
    #  cv2.waitKey(0)
    #  cv2.destroyAllWindows()
