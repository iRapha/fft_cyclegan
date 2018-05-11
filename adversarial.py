import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# first, figure out which image is 1005_A:
input_goal_A = dataset.dataset.get_single('./datasets/maps/testA/1005_A.jpg', './datasets/maps/testB/1005_B.jpg')['A']

# optimize 1005_B and 1018_B to reconstruct 1005_A.
template = './datasets/maps/test{0}/{1}_{0}.jpg'
for i, data_id in enumerate(['1005', '1018']):
    data = dataset.dataset.get_single(template.format('A', data_id), template.format('B', data_id))
    model.set_input(data)
    visuals = model.adversarial(input_goal_A)
    img_path = model.get_image_paths()
    print('%04d: process image... %s' % (i, img_path))
    visualizer.save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio)

webpage.save()
