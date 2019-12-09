#!/usr/bin/env python3

# Downloaded from:
# https://github.com/bioinf-jku/TTUR/blob/master/precalc_stats_example.py

import os
import glob
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
import fid
from scipy.misc import imread
import tensorflow as tf

import datasets as dset

########
# PATHS
########
data_root = 'data' # set path to training set images
image_size = 48
output_path = 'stl10_size_%i_fid_stats.npz' % image_size # path for where to store the statistics
# if you have downloaded and extracted
#   http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
# set this path to the directory where the extracted files are, otherwise
# just set it to None and the script will later download the files for you
inception_path = None
print("check for inception model..", end=" ", flush=True)
inception_path = fid.check_or_download_inception(inception_path) # download inception if necessary
print("ok")

# loads all images into memory (this might require a lot of RAM!)
print("load images..", end=" " , flush=True)
    
which_dataset = dset.STL10

train_transform = [transforms.RandomCrop(90),
                           transforms.Resize(image_size),
                           transforms.RandomHorizontalFlip()]

unl_set = which_dataset(root=data_root, transform=train_transform,
                        load_in_mem=False, train=False)
theloader = DataLoader(unl_set, batch_size=50, shuffle=True)

for xy in theloader:
  x, y = xy



image_list = glob.glob(os.path.join(data_path, '*.jpg'))
images = np.array([imread(str(fn)).astype(np.float32) for fn in image_list])
print("%d images found and loaded" % len(images))

print("create inception graph..", end=" ", flush=True)
fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
print("ok")

print("calculte FID stats..", end=" ", flush=True)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    mu, sigma = fid.calculate_activation_statistics(images, sess, batch_size=100)
    np.savez_compressed(output_path, mu=mu, sigma=sigma)
print("finished")
