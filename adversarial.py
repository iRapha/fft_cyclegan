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
for i, data in enumerate(dataset):
    if '1005' in data['A_paths']:
        input_goal_A = data['A']

# optimize 1005_B and 1018_B to reconstruct 1005_A.
for i, data in enumerate(dataset):
    if '1005' not in data['A_paths'] and '1018' not in data['A_paths']:
        continue

    model.set_input(data)
    visuals = model.adversarial(input_goal_A)
    img_path = model.get_image_paths()
    print('%04d: process image... %s' % (i, img_path))
    visualizer.save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio)

webpage.save()
