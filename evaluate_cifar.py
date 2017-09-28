# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 13:02:07 2016

@author: souvenir
"""

import tensorflow as tf
from classfile import CombinatorialTripletSet
import os
import time
import numpy as np
from PIL import Image
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets import alexnet

filename = './inputs/cifar/test.txt'
pretrained_net = './output/cifar/no_triplets/ckpts/checkpoint-4999'
img_size = [256, 256]
crop_size = [224, 224]
featLayer = 'alexnet_v2/fc7'
mean_file = './models/cifar/cifar_mean_im.npy'

batch_size = 30
num_pos_examples = batch_size/10

image_batch = tf.placeholder(tf.float32, shape=[batch_size, crop_size[0], crop_size[0], 3])
label_batch = tf.placeholder(tf.int32, shape=(batch_size))

print("Preparing network...")
with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
    _, layers = alexnet.alexnet_v2(image_batch, num_classes=100, is_training=False)

feat = tf.squeeze(layers[featLayer])

# Create a saver for writing loading checkpoints.
saver = tf.train.Saver()

# Create data "batcher"
data = CombinatorialTripletSet(filename, mean_file, img_size, crop_size, batch_size, num_pos_examples, isTraining=False)

c = tf.ConfigProto()
c.gpu_options.visible_device_list="2,3"

sess = tf.Session(config=c)
# Here's where we need to load saved weights
saver.restore(sess, pretrained_net)

allFeats = []
allLabels = []
allIms = []
num_iters = np.sum([len(data.files[ix]) for ix in range(0,len(data.files))]) / batch_size

for step in range(num_iters):
    print float(step)/float(num_iters)
    start_time = time.time()
    batch, labels, ims = data.getBatch()
    f = sess.run(feat, feed_dict={image_batch: batch})
    for ix in range(0,len(ims)):
        im = ims[ix]
        if im not in allIms:
            allIms.append(im)
            allFeats.append(f[ix])
            allLabels.append(labels[ix])
    duration = time.time() - start_time

def getDist(feat,otherFeats):
    dist = (otherFeats - feat)**2
    dist = np.sum(dist,axis=1)
    dist = np.sqrt(dist)
    return dist

def combine_horz(ims):
    images = map(Image.open, [ims[0],ims[1],ims[2],ims[3],ims[4],ims[5],ims[6]])
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    return new_im

npAllFeats = np.array(allFeats)
npAllLabels = np.array(allLabels)
success = np.zeros((len(npAllFeats),100))
ctr = 0
numIms= len(allIms)
for im,feat,cls in zip(allIms,npAllFeats,npAllLabels):
    print ctr, numIms
    dists = getDist(feat,npAllFeats)
    sortInds = np.argsort(dists)
    hits = np.where(npAllLabels[sortInds]==cls)[0][1:]
    topHit = np.min(hits)-1
    if topHit < 100:
        success[ctr,topHit:] = 1
    ctr += 1

print np.mean(success,axis=0)

# output_dir = '/Users/abby/Desktop/cifar_results'
# if os.path.exists(output_dir):
#     os.rmdir(output_dir)
#
# good_dir = os.path.join(output_dir,'good')
# bad_dir = os.path.join(output_dir,'bad')
#
# os.makedirs(output_dir)
# os.makedirs(good_dir)
# os.makedirs(bad_dir)
#
# npAllFeats = np.array(allFeats)
# npAllLabels = np.array(allLabels)
# success = np.zeros((len(npAllFeats),100))
# ctr = 0
# for im,feat,cls in zip(allIms,npAllFeats,npAllLabels):
#     dists = getDist(feat,npAllFeats)
#     sortInds = np.argsort(dists)
#     hits = np.where(npAllLabels[sortInds]==cls)[0][1:]
#     topHit = np.min(hits)-1
#     topHitIm = allIms[sortInds[hits[0]]]
#     topMatchIm1 = allIms[sortInds[1]]
#     topMatchIm2 = allIms[sortInds[2]]
#     topMatchIm3 = allIms[sortInds[3]]
#     topMatchIm4 = allIms[sortInds[4]]
#     topMatchIm5 = allIms[sortInds[5]]
#     new_im = combine_horz([im,topMatchIm1,topMatchIm2,topMatchIm3,topMatchIm4,topMatchIm5,topHitIm])
#     if topHit < 100:
#         print 'Good ', topHit
#         print im, topHitIm
#         save_path = os.path.join(good_dir,str(ctr)+'_'+str(topHit)+'.jpg')
#         success[ctr,topHit:] = 1
#     else:
#         save_path = os.path.join(bad_dir,str(ctr)+'_'+str(topHit)+'.jpg')
#         print 'Bad ', topHit
#     new_im.save(save_path)
#     ctr += 1
#
# np.mean(success,axis=0)
