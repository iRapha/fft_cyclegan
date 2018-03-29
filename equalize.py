import cv2
import numpy as np
from util.util import save_image
import os
import argparse

def get_paths(source, recurse=False):
    paths = []
    names = []
    for _, _, filenames in os.walk(source):
        for i, f in enumerate(filenames):
            names.append(f)
            paths.append(os.path.join(source, f))
        if not recurse:
            break

    return paths, names

def equalize_img(filename, clip=40):
    img = cv2.imread(filename,1)
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
    # cl1 = clahe.apply(r)
    temp = np.zeros(img.shape, dtype=np.uint8)
    temp[:, :, 0] = clahe.apply(r)
    temp[:, :, 1] = clahe.apply(g)
    temp[:, :, 2] = clahe.apply(b)
    return temp[..., [2, 1, 0]]

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Equalize images")
    arg_parser.add_argument('--source-dir', required=True, help='path to sources', type=str)
    arg_parser.add_argument('--dest-dir', required=True, help='path to destination', type=str)
    # save_image(equalize_img('1005_A_fake_B.png'), 'test.png')
    args = arg_parser.parse_args()
    paths, names = get_paths(args.source_dir)
    for path, name in zip(paths, names):
        print(name)
        out_img = equalize_img(path)
        out_filename = 'equalized_{}'.format(name)
        out_path = os.path.join(args.dest_dir, out_filename)
        save_image(out_img, out_path)  
    
