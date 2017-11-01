# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 13:02:07 2016

@author: souvenir
"""

import tensorflow as tf
from classfile import CombinatorialTripletSet
import os.path
import time
import numpy as np
from PIL import Image
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets import resnet_v2

filename = './inputs/traffickcam/test_equal.txt'
pretrained_net = './output/traffickcam/ckpts/checkpoint-201710311223_lr0pt0005_outputSz128_margin0pt3-31499'
img_size = [256, 256]
crop_size = [227, 227]
# featLayer = 'alexnet_v2/fc7'
mean_file = './models/places365/places365CNN_mean.npy'

batch_size = 30
num_pos_examples = batch_size/10

output_size = 100

image_batch = tf.placeholder(tf.float32, shape=[batch_size, crop_size[0], crop_size[0], 3])
label_batch = tf.placeholder(tf.int32, shape=(batch_size))

print("Preparing network...")
with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    _, layers = resnet_v2.resnet_v2_50(image_batch, num_classes=output_size, is_training=True)

feat = tf.squeeze(tf.nn.l2_normalize(tf.get_default_graph().get_tensor_by_name("resnet_v2_50/pool5:0"),3))

# Create a saver for writing loading checkpoints.
saver = tf.train.Saver()

# Create data "batcher"
data = CombinatorialTripletSet(filename, mean_file, img_size, crop_size, batch_size, num_pos_examples, isTraining=False)

# c = tf.ConfigProto()
# c.gpu_options.visible_device_list="3"

# sess = tf.Session(config=c)
sess = tf.Session()
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

good_dir = '/project/focus/abby/triplepalooza/example_results/good'
bad_dir = '/project/focus/abby/triplepalooza/example_results/bad'

npAllFeats = np.array(allFeats)
npAllLabels = np.array(allLabels)
success = np.zeros((len(npAllFeats),100))
ctr = 0
for im,feat,cls in zip(allIms,npAllFeats,npAllLabels):
    dists = getDist(feat,npAllFeats)
    sortInds = np.argsort(dists)
    hits = np.where(npAllLabels[sortInds]==cls)[0][1:]
    topHit = np.min(hits)-1
    topHitIm = allIms[sortInds[hits[0]]]
    topMatchIm1 = allIms[sortInds[1]]
    topMatchIm2 = allIms[sortInds[2]]
    topMatchIm3 = allIms[sortInds[3]]
    topMatchIm4 = allIms[sortInds[4]]
    topMatchIm5 = allIms[sortInds[5]]
    new_im = combine_horz([im,topMatchIm1,topMatchIm2,topMatchIm3,topMatchIm4,topMatchIm5,topHitIm])
    if topHit < 100:
        print 'Good ', topHit
        print im, topHitIm
        save_path = os.path.join(good_dir,str(ctr)+'_'+str(topHit)+'.jpg')
        success[ctr,topHit:] = 1
    else:
        save_path = os.path.join(bad_dir,str(ctr)+'_'+str(topHit)+'.jpg')
        print 'Bad ', topHit
    new_im.save(save_path)
    ctr += 1

np.mean(success,axis=0)
