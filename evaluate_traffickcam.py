# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 13:02:07 2016

@author: souvenir
"""

import tensorflow as tf
from classfile import CombinatorialTripletSet
import os.path
import time
from alexnet import CaffeNetPlaces365
import numpy as np

filename = './inputs/traffickcam/test.txt'
checkpoint_file = './output/ckpts/checkpoint-9599'
img_size = [256, 256]
crop_size = [227, 227]
featLayer = 'fc7'
mean_file = './models/places365/places365CNN_mean.npy'

batch_size = 200
num_pos_examples = batch_size/10

image_batch = tf.placeholder(tf.float32, shape=[batch_size, crop_size[0], crop_size[0], 3])
label_batch = tf.placeholder(tf.int32, shape=(batch_size))

print("Preparing network...")
net = CaffeNetPlaces365({'data': image_batch})
feat = net.layers[featLayer]

# Create a saver for writing loading checkpoints.
saver = tf.train.Saver()

# Create data "batcher"
data = CombinatorialTripletSet(filename, mean_file, img_size, crop_size, batch_size, num_pos_examples)

sess = tf.Session()
# Here's where we need to load saved weights
saver.restore(sess, checkpoint_file)

allFeats = []
allLabels = []
num_iters = np.sum([len(data.files[ix]) for ix in range(0,len(data.files))]) / batch_size

for step in range(num_iters):
    start_time = time.time()
    batch, labels = data.getBatch()
    f = sess.run(feat, feed_dict={image_batch: batch})
    allFeats.extend(f)
    allLabels.extend(labels)
    duration = time.time() - start_time

def getDist(feat,otherFeats):
    dist = (otherFeats - feat)**2
    dist = np.sum(dist,axis=1)
    dist = np.sqrt(dist)
    return dist

npAllFeats = np.array(allFeats)
npAllLabels = np.array(allLabels)
success = np.zeros((len(npAllFeats),100))
ctr = 0
for feat,cls in zip(npAllFeats,npAllLabels):
    dists = getDist(feat,npAllFeats)
    sortInds = np.argsort(dists)
    hits = np.where(npAllLabels[sortInds]==cls)[0][1:]
    topHit = np.min(hits)-1
    if topHit < 100:
        success[ctr,topHit:] = 1
    ctr += 1

np.mean(success,axis=0)
