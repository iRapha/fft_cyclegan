import os
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
    base_dir = '1005s'
    for exp in os.listdir(base_dir):
        for img_name in os.listdir(os.path.join(base_dir, exp)):
            if 'fake_B' in img_name or 'real_B' in img_name:
                new_name = img_name.replace('fake_B', 'fakeviz_B').replace('real_B', 'realviz_B')
                og = os.path.join(base_dir, exp, img_name)
                nu = os.path.join(base_dir, exp, new_name)
                save_image(equalize_img(og), nu)

