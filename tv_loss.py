import torch
import torch.autograd as ag
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from util.util import get_paths, load_image
import numpy as np
import argparse
import os

def TV(X):
    tv_x = torch.sum(torch.abs(X[..., :-1, :] - X[..., 1:, :]))
    tv_y = torch.sum(torch.abs(X[..., :-1] - X[..., 1:]))
    return tv_x + tv_y

pipeline = transforms.Compose([transforms.ToTensor(),
                            transforms.Lambda(lambda x : x.mul(255))])
def get_image_tensor(path):
    image = load_image(path)
    # batch = pipeline(image).unsqueeze(0)
    return pipeline(image)

def write_image_tensor(path, data):
    data = data.clone().clamp(0, 255).numpy()
    image = Image.fromarray(data.transpose(1, 2, 0).astype("uint8") )
    image.save(path)

def optimize(image_tensor, target_loss = 8.5e5, lr = 0.001, loss_interval=50):
    loss = None
    loss_check = None
    image = nn.Parameter(image_tensor.clone())
    optimizer = torch.optim.SGD([image], lr = lr)

    it = 0
    
    while True:
        loss = TV(image)
        loss.backward()
        optimizer.step()
        l = loss.data[0]
        if it % 10 == 0:
            print('Iteration {} Loss {}'.format(it, l))
            loss_check = loss.data[0]
        
        if it % loss_interval == 0:
            loss_check = l

        if l < target_loss:
            break

        if loss_check is not None and loss_check < l:
            print("{} > {}".format(l, loss_check))
            print("Loss has increased, stopping early at {}".format(l))
            break
        it += 1
    print("Done at iter {} with loss {}".format(it, l))
    return image.data

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
                    description = "Optimize TV loss of images in a directory")
    arg_parser.add_argument("--source-dir", type=str, required=True,
                            help="path to image source directory")
    arg_parser.add_argument("--dest-dir", type=str, required=True,
                            help="path to image destination directory")
    arg_parser.add_argument("--target-tv", type=float, default=8.5e5,
                            help="Desired TV loss to optimize to")
    arg_parser.add_argument("--lr", type=float, default=1e-3,
                            help="SGD learning rate for optimization")
    arg_parser.add_argument("--filter", type=int, default=1,
                            help="1 to only run on *fake_B*, 0 to run on all images ")

    args = arg_parser.parse_args()

    paths, names = get_paths(args.source_dir, recurse = True)
    source_images = list(zip(paths, names))
    if args.filter == 1:
        source_images = [(p, i) for p, i in source_images if 'fake_B' in i and 'norm_' not in i]
    
    for path, name in source_images:
        print('Optimizing {}'.format(path))
        image = get_image_tensor(path)
        _, subdir_name = os.path.split(os.path.split(path)[0])
        optimize(image, lr=args.lr)

        #strip file ending from image name
        image_name = '.'.join(name.split('.')[:-1])
        ext = name.split('.')[-1]
        image_name = '{}_tv_optmized.{}'.format(subdir_name, ext)
        write_image_tensor(os.path.join(args.dest_dir, image_name), image)
    

    # sources = '/home/marc/datasets/maps/1005s'
    # source_images = list(zip(*get_paths(sources, recurse=True)))
    # source_images = [(p, i) for p, i in source_images if 'real_B' in i or 'fake_B' in i]

    # path, name = source_images[0]
    # image = get_image_tensor(path)
    
    # print(path)
    # print(TV(image))
    # image = optimize(image)
    # print(TV(image))
    # write_image_tensor('temp/tv.test.png', image)

    # image = nn.Parameter(image)
    # loss = 1e15
    # while l
        # optim.zero_grad()
        # loss = TV(image)
        # print(loss.data)
        # loss.backward()
        # optim.step()

    # print(path)
    # write_image_tensor('temp.tv.test.png', image.data)

    # for path, name in source_images:
        # batch = get_image_tensor(path)
        # tv = TV(batch)
        # print('{} has TV loss {}'.format(path, tv))


# A = torch.LongTensor([[[1, 2], [1, 1]], [[2, 1,], [1, 1]]])
# print(A)
# print(TV(A))
