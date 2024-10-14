import numpy as np
import os
import torch
import torch.nn as nn

from models import create_model
from data import create_dataset
from options.test_options import TestOptions
from util.util import tensor2im
import cv2
opt = TestOptions().parse()  # get test options
# hard-code some parameters for test
opt.num_threads = 0   # test code only supports num_threads = 1
opt.batch_size = 1    # test code only supports batch_size = 1
opt.serial_batches = False  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)
model.eval()
for i, data in enumerate(dataset):
    model.set_input(data)  # unpack data from data loader
    model.test()           # run inference
    img_path = model.get_image_paths()
    img_name = img_path[0].split('/')[-1]
    print(img_name)
    output = tensor2im(model.fake_B)
    w, h = data['original_size']
    print(w,h)
    img_output = output[:h, :w, :]
    output_path = './results_C92/{}_epoch_{}'.format(opt.name,opt.epoch)
    os.makedirs(output_path,exist_ok=True)
    cv2.imwrite(os.path.join(output_path,img_name), img_output)

