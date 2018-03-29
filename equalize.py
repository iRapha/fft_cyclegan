import cv2
import numpy as np
from util.util import save_image
def equalize_img(filename, clip=40):
    img = cv2.imread(filename,1)
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
    # cl1 = clahe.apply(r)
    temp = np.zeros((256, 256, 3), dtype=np.uint8)
    temp[:, :, 0] = clahe.apply(r)
    temp[:, :, 1] = clahe.apply(g)
    temp[:, :, 2] = clahe.apply(b)
    return temp[..., [2, 1, 0]]

if __name__ == '__main__':
    save_image(equalize_img('1005_A_fake_B.png'), 'test.png')
    
