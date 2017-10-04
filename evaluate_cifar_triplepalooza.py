import tensorflow as tf
from classfile import NonTripletSet
import os
import time
import numpy as np
from PIL import Image
import scipy.spatial.distance
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets import resnet_v2
import random

def getDist(feat,otherFeats):
    # dist = (otherFeats - feat)**4
    # dist = np.sum(dist,axis=1)
    # dist = np.power(dist,.25)
    dist = (otherFeats - feat)**2
    dist = np.sum(dist,axis=1)
    dist = np.sqrt(dist)
    return dist

train_file = './inputs/cifar/train.txt'
test_file = './inputs/cifar/test.txt'
train_net = './output/cifar/no_doctoring/ckpts/checkpoint-6999'
test_net = './output/cifar/no_doctoring/ckpts/checkpoint-6999'
img_size = [256, 256]
crop_size = [224, 224]
featLayer = 'resnet_v2_50/logits'
mean_file = './models/cifar/cifar_mean_im.npy'

batch_size = 90
num_pos_examples = batch_size/10

# Create train_data "batcher"
train_data = NonTripletSet(train_file, mean_file, img_size, crop_size, batch_size, num_pos_examples)
test_data = NonTripletSet(test_file, mean_file, img_size, crop_size, batch_size, num_pos_examples)

image_batch = tf.placeholder(tf.float32, shape=[batch_size, crop_size[0], crop_size[0], 3])
repMeanIm = np.tile(np.expand_dims(train_data.meanImage,0),[batch_size,1,1,1])
noise = tf.random_normal(shape=[batch_size, crop_size[0], crop_size[0], 3], mean=0.0, stddev=3, dtype=tf.float32)
final_batch = tf.add(tf.subtract(image_batch,repMeanIm),noise)
label_batch = tf.placeholder(tf.int32, shape=(batch_size))

print("Preparing network...")
with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    _, layers = resnet_v2.resnet_v2_50(final_batch, num_classes=100, is_training=True)

feat = tf.squeeze(layers[featLayer])

c = tf.ConfigProto()
c.gpu_options.visible_device_list="0,1"

# TESTING ACCURACY
sess = tf.Session(config=c)
saver = tf.train.Saver(max_to_keep=100)
saver.restore(sess, train_net)

trainingImsAndLabels = [(train_data.files[ix][iy],train_data.classes[ix]) for ix in range(len(train_data.files)) for iy in range(len(train_data.files[ix]))]
random.shuffle(trainingImsAndLabels)
trainingImsAndLabels = trainingImsAndLabels[:10000]
numTrainingIms = len(trainingImsAndLabels)
trainingFeats = np.empty((numTrainingIms,feat.shape[1]),dtype=np.float32)
trainingIms = np.empty((numTrainingIms),dtype=object)
trainingLabels = np.empty((numTrainingIms),dtype=np.int32)
num_iters = numTrainingIms / batch_size

print 'Computing training set features...'
for step in range(0,num_iters+1):
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
    sortedInds = np.argsort(dists)[1:]
    sortedLabels = trainingLabels[sortedInds][:100]
    if thisLabel in sortedLabels:
        topHit = np.where(sortedLabels==thisLabel)[0][0]
        trainingAccuracy[idx,topHit:] = 1
    # if idx%10==0:
    #     print idx,': ',np.mean(trainingAccuracy[:idx,:],axis=0)[0]

sess.close()

# TESTING ACCURACY
sess = tf.Session(config=c)
saver = tf.train.Saver(max_to_keep=100)
saver.restore(sess, test_net)

testingImsAndLabels = [(test_data.files[ix][iy],test_data.classes[ix]) for ix in range(len(test_data.files)) for iy in range(len(test_data.files[ix]))]
random.shuffle(testingImsAndLabels)
testingImsAndLabels = testingImsAndLabels[:10000]
numTestingIms = len(testingImsAndLabels)
testingFeats = np.empty((numTestingIms,feat.shape[1]),dtype=np.float32)
testingIms = np.empty((numTestingIms),dtype=object)
testingLabels = np.empty((numTestingIms),dtype=np.int32)
num_iters = numTestingIms / batch_size

print 'Computing testing set features...'
for step in range(0,num_iters+1):
    # print step, '/', num_iters
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
    dists = getDist(thisFeat,testingFeats)
    sortedInds = np.argsort(dists)[1:]
    sortedLabels = testingLabels[sortedInds][:100]
    if thisLabel in sortedLabels:
        topHit = np.where(sortedLabels==thisLabel)[0][0]
        testingAccuracy[idx,topHit:] = 1
    # if idx%10==0:
    #     print idx,': ',np.mean(testingAccuracy[:idx,:],axis=0)[0]

sess.close()

print '---Triplepalooza--'
print 'Network: ', test_net
print 'Top1 Training Accuracy: ', float(trainingTop1Accuracy)/float(numTrainingIms)
print 'NN Training Accuracy: ',np.mean(trainingAccuracy,axis=0)
print '---'
print 'Top1 Test Accuracy: ', float(testingTop1Accuracy)/float(numTestingIms)
print 'NN Test Accuracy: ',np.mean(testingAccuracy,axis=0)
print '---'
