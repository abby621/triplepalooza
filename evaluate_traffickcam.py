# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 13:02:07 2016

@author: souvenir
"""

import tensorflow as tf
from classfile import NonTripletSet
import os.path
import time
import numpy as np
from PIL import Image
import random
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets import resnet_v2

def getDist(feat,otherFeats):
    dist = [np.dot(feat,otherFeat) for otherFeat in otherFeats]
    return dist

train_file = './inputs/traffickcam/train_equal.txt'
test_file = './inputs/traffickcam/test_equal.txt'
pretrained_net = './output/traffickcam/ckpts/checkpoint-201710311223_lr0pt0005_outputSz128_margin0pt3-31499'
img_size = [256, 256]
crop_size = [227, 227]
# featLayer = 'alexnet_v2/fc7'
mean_file = './models/places365/places365CNN_mean.npy'

batch_size = 100
num_pos_examples = batch_size/10

output_size = 128

image_batch = tf.placeholder(tf.float32, shape=[batch_size, crop_size[0], crop_size[0], 3])
label_batch = tf.placeholder(tf.int32, shape=(batch_size))

print("Preparing network...")
with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    _, layers = resnet_v2.resnet_v2_50(image_batch, num_classes=output_size, is_training=True)

feat = tf.squeeze(tf.nn.l2_normalize(tf.get_default_graph().get_tensor_by_name("resnet_v2_50/pool5:0"),3))

# Create a saver for writing loading checkpoints.
saver = tf.train.Saver()

# Create data "batcher"
#image_list, mean_file, image_size, crop_size, batchSize=100, isTraining=True
train_data = NonTripletSet(train_file, mean_file, img_size, crop_size, batch_size, isTraining=False)
test_data = NonTripletSet(test_file, mean_file, img_size, crop_size, batch_size, isTraining=False)

# c = tf.ConfigProto()
# c.gpu_options.visible_device_list="3"

# sess = tf.Session(config=c)
sess = tf.Session()
# Here's where we need to load saved weights
saver.restore(sess, pretrained_net)

trainingImsAndLabels = [(train_data.files[ix][iy],train_data.classes[ix]) for ix in range(len(train_data.files)) for iy in range(len(train_data.files[ix]))]
random.shuffle(trainingImsAndLabels)
trainingImsAndLabels = trainingImsAndLabels[:10000]
numTrainingIms = len(trainingImsAndLabels)
trainingFeats = np.empty((numTrainingIms,feat.shape[1]),dtype=np.float32)
trainingIms = np.empty((numTrainingIms),dtype=object)
trainingLabels = np.empty((numTrainingIms),dtype=np.int32)
num_iters = numTrainingIms / batch_size

print 'Computing training set features...'
for step in range(0,num_iters):
    print step, '/', num_iters
    if step == num_iters:
        end_ind = numTrainingIms
    else:
        end_ind = step*batch_size+batch_size

    il = trainingImsAndLabels[step*batch_size:end_ind]
    ims = [i[0] for i in il]
    trainingIms[step*batch_size:end_ind] = ims
    labels = [i[1] for i in il]
    trainingLabels[step*batch_size:end_ind] = labels
    batch = train_data.getBatchFromImageList(ims)

    while len(labels) < batch_size:
        labels += [labels[-1]]
        batch = np.vstack((batch,np.expand_dims(batch[-1],0)))

    ff = sess.run(feat, feed_dict={image_batch: batch, label_batch:labels})
    trainingFeats[step*batch_size:end_ind,:] = ff[:len(il),:]

print 'Computing training set distances...'
trainingAccuracy = np.zeros((numTrainingIms,100))
for idx in range(numTrainingIms):
    thisFeat = trainingFeats[idx,:]
    thisLabel = trainingLabels[idx]
    dists = getDist(thisFeat,trainingFeats)
    sortedInds = np.argsort(dists)
    sortedLabels = trainingLabels[sortedInds][:100]
    if thisLabel in sortedLabels:
        topHit = np.where(sortedLabels==thisLabel)[0][0]
        trainingAccuracy[idx,topHit:] = 1
    if idx%10==0:
        print idx,': ',np.mean(trainingAccuracy[:idx,:],axis=0)[0]

sess.close()

print '---Triplepalooza--'
print 'Network: ', eval_net
print 'NN Training Accuracy: ',np.mean(trainingAccuracy,axis=0)

# TESTING ACCURACY

# save out images
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

sess = tf.Session(config=c)
saver = tf.train.Saver(max_to_keep=100)
saver.restore(sess, eval_net)

testingImsAndLabels = [(test_data.files[ix][iy],test_data.classes[ix]) for ix in range(len(test_data.files)) for iy in range(len(test_data.files[ix]))]
random.shuffle(testingImsAndLabels)
testingImsAndLabels = testingImsAndLabels[:10000]
numTestingIms = len(testingImsAndLabels)
testingFeats = np.empty((numTestingIms,feat.shape[1]),dtype=np.float32)
testingIms = np.empty((numTestingIms),dtype=object)
testingLabels = np.empty((numTestingIms),dtype=np.int32)
num_iters = numTestingIms / batch_size

print 'Computing testing set features...'
for step in range(0,num_iters):
    print step, '/', num_iters
    if step == num_iters:
        end_ind = numTestingIms
    else:
        end_ind = step*batch_size+batch_size

    il = testingImsAndLabels[step*batch_size:end_ind]
    ims = [i[0] for i in il]
    testingIms[step*batch_size:end_ind] = ims
    labels = [i[1] for i in il]
    testingLabels[step*batch_size:end_ind] = labels
    batch = test_data.getBatchFromImageList(ims)

    while len(labels) < batch_size:
        labels += [labels[-1]]
        batch = np.vstack((batch,np.expand_dims(batch[-1],0)))

    ff = sess.run(feat, feed_dict={image_batch: batch, label_batch:labels})
    testingFeats[step*batch_size:end_ind,:] = ff[:len(il),:]

print 'Computing testing set distances...'
testingAccuracy = np.zeros((numTestingIms,100))
for idx in range(numTestingIms):
    thisFeat = testingFeats[idx,:]
    thisLabel = testingLabels[idx]
    thisCam = testingCams[idx]
    dists = getDist(thisFeat,testingFeats)
    sortedInds = np.argsort(dists)
    sortedLabels = testingLabels[sortedInds]

    topHit = np.where(sortedLabels==thisLabel)[0][0]
    topHitIm = testingIms[sortedInds[topHit]]
    topMatchIm1 = testingIms[sortedInds[0]]
    topMatchIm2 = testingIms[sortedInds[1]]
    topMatchIm3 = testingIms[sortedInds[2]]
    topMatchIm4 = testingIms[sortedInds[3]]
    topMatchIm5 = testingIms[sortedInds[4]]
    new_im = combine_horz([im,topMatchIm1,topMatchIm2,topMatchIm3,topMatchIm4,topMatchIm5,topHitIm])

    if thisLabel in sortedLabels[:100]:
        testingAccuracy[idx,topHit:] = 1
        save_path = os.path.join(good_dir,str(ctr)+'_'+str(topHit)+'.jpg')
    else:
        save_path = os.path.join(bad_dir,str(ctr)+'_'+str(topHit)+'.jpg')

    new_im.save(save_path)

    if idx%10==0:
        print idx,': ',np.mean(testingAccuracy[:idx,:],axis=0)[0]

sess.close()

print '---Triplepalooza--'
print 'Network: ', eval_net
print 'NN Training Accuracy: ',np.mean(trainingAccuracy,axis=0)
print '---'
print 'NN Test Accuracy: ',np.mean(testingAccuracy,axis=0)
print '---'

# allFeats = []
# allLabels = []
# allIms = []
# num_iters = np.sum([len(data.files[ix]) for ix in range(0,len(data.files))]) / batch_size
#
# for step in range(num_iters):
#     print float(step)/float(num_iters)
#     start_time = time.time()
#     batch, labels, ims = data.getBatch()
#     f = sess.run(feat, feed_dict={image_batch: batch})
#     for ix in range(0,len(ims)):
#         im = ims[ix]
#         if im not in allIms:
#             allIms.append(im)
#             allFeats.append(f[ix])
#             allLabels.append(labels[ix])
#     duration = time.time() - start_time
#
# def getDist(feat,otherFeats):
#     dist = (otherFeats - feat)**2
#     dist = np.sum(dist,axis=1)
#     dist = np.sqrt(dist)
#     return dist
#
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
