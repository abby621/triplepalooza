# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 13:02:07 2016

@author: souvenir
"""

import tensorflow as tf
from classfile import NonTripletSet
import os
import time
import numpy as np
from PIL import Image
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets import alexnet

filename = './inputs/cifar/test.txt'
pretrained_net = './output/cifar/no_triplets/ckpts/checkpoint-49999'
img_size = [256, 256]
crop_size = [224, 224]
featLayer = 'alexnet_v2/fc8'
mean_file = './models/cifar/cifar_mean_im.npy'

batch_size = 100
num_pos_examples = batch_size/10

data = NonTripletSet(filename, mean_file, img_size, crop_size, batch_size, isTraining=False)

image_batch = tf.placeholder(tf.float32, shape=[batch_size, crop_size[0], crop_size[0], 3])
repMeanIm = np.tile(np.expand_dims(data.meanImage,0),[batch_size,1,1,1])
final_batch = image_batch - repMeanIm
label_batch = tf.placeholder(tf.int32, shape=(batch_size))

print("Preparing network...")
with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
    _, layers = alexnet.alexnet_v2(final_batch, num_classes=100, is_training=False)

feat = tf.squeeze(layers[featLayer])
f2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=feat, labels=label_batch)

# Create a saver for writing loading checkpoints.
saver = tf.train.Saver()

c = tf.ConfigProto()
c.gpu_options.visible_device_list="2,3"

sess = tf.Session(config=c)
# Here's where we need to load saved weights
saver.restore(sess, pretrained_net)

allFeats = []
allLabels = []
allIms = []
num_iters = np.sum([len(data.files[ix]) for ix in range(0,len(data.files))]) / batch_size

imsAndLabels = [(data.files[ix][iy],data.classes[ix]) for ix in range(len(data.files)) for iy in range(len(data.files[ix]))]
num_iters = len(imsAndLabels) / batch_size

top1 = 0.0
for step in range(0,num_iters):
    start_time = time.time()
    il = imsAndLabels[step*batch_size:step*batch_size+batch_size]
    ims = [ii[0] for ii in il]
    labels = [ii[1] for ii in il]
    batch = data.getBatchFromImageList(ims)
    ff = sess.run(f2, feed_dict={image_batch: batch, label_batch:labels})
    topHit = len(np.where(ff==0)[0])
    top1 += topHit
    print float(step)/float(num_iters), top1/((step+1)*batch_size)
    duration = time.time() - start_time
