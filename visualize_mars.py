import tensorflow as tf
from classfile import NonTripletSet,CombinatorialTripletSet
import os
import time
from datetime import datetime
import numpy as np
from PIL import Image
import scipy.spatial.distance
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets import resnet_v2
import random
from scipy.ndimage import zoom

def getDist(feat,otherFeat):
    return np.dot(feat,otherFeat)

train_file = './inputs/mars/train.txt'
test_file = './inputs/mars/test.txt'
eval_net = './output/mars/ckpts/final-_lr0pt0001_outputSz100_margin0.3-19999'
img_size = [256, 256]
crop_size = [224, 224]
featLayer = 'resnet_v2_50/logits'
convOutLayer = 'resnet_v2_50/block4'
mean_file = './models/mars/mean_im.npy'

batch_size = 100
num_pos_examples = batch_size/10

# Create train_data "batcher"
train_data = NonTripletSet(train_file, mean_file, img_size, crop_size, batch_size, num_pos_examples)
test_data = NonTripletSet(test_file, mean_file, img_size, crop_size, batch_size, num_pos_examples)

image_batch = tf.placeholder(tf.float32, shape=[batch_size, crop_size[0], crop_size[0], 3])
repMeanIm = np.tile(np.expand_dims(train_data.meanImage,0),[batch_size,1,1,1])
final_batch = tf.subtract(image_batch,repMeanIm)
label_batch = tf.placeholder(tf.int32, shape=(batch_size))

print("Preparing network...")
with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    _, layers = resnet_v2.resnet_v2_50(final_batch, num_classes=100, is_training=True)

feat = tf.nn.l2_normalize(layers[featLayer],3)
convOut = tf.nn.l2_normalize(layers[convOutLayer],3)
weights = tf.squeeze(tf.get_default_graph().get_tensor_by_name("resnet_v2_50/logits/weights:0"))
gap = tf.squeeze(tf.get_default_graph().get_tensor_by_name("resnet_v2_50/pool5:0"))

featInd = tf.placeholder(tf.int32, shape=())
saliency_map = tf.gradients(tf.gather(tf.squeeze(feat),featInd), final_batch)[0]

filters = tf.gradients(tf.gather(tf.squeeze(convOut),featInd), final_batch)[0]

c = tf.ConfigProto()
c.gpu_options.visible_device_list="0,1"
sess = tf.Session(config=c)
saver = tf.train.Saver(max_to_keep=100)
saver.restore(sess, eval_net)

logs_path = '/tmp/tensorflow/'

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(logs_path,sess.graph)

trainingImsAndLabels = [(train_data.files[ix][iy],train_data.classes[ix]) for ix in range(len(train_data.files)) for iy in range(len(train_data.files[ix]))]
random.shuffle(trainingImsAndLabels)
trainingImsAndLabels = trainingImsAndLabels[:5000]
numTrainingIms = len(trainingImsAndLabels)

trainingLabels = np.array([label for (im,label) in trainingImsAndLabels])
indsByLabel = {}
for label in np.unique(trainingLabels):
    goodInds = np.where(trainingLabels==label)[0]
    if len(goodInds) > 1:
        indsByLabel[label] = goodInds

reppedLabels = np.array(indsByLabel.keys())

trainingFeats = np.empty((numTrainingIms,feat.shape[3]),dtype=np.float32)
trainingIms = np.empty((numTrainingIms),dtype=object)
trainingLabels = np.empty((numTrainingIms),dtype=np.int32)
for idx in range(0,numTrainingIms,batch_size):
    print idx, '/', numTrainingIms
    il = trainingImsAndLabels[idx:idx+batch_size]
    ims = [i[0] for i in il]
    labels = [i[1] for i in il]
    batch = train_data.getBatchFromImageList(ims)
    trainingIms[idx:idx+batch_size] = ims
    trainingLabels[idx:idx+batch_size] = labels
    ff = sess.run(feat, feed_dict={image_batch: batch, label_batch:labels})
    trainingFeats[idx:idx+batch_size,:] = np.squeeze(ff)

# def combine_horz(images):
#     new_im = np.zeros((images[0].shape[0],images[0].shape[1]*3,3))
#     new_im[:images[0].shape[1],:images[0].shape[0],0] = images[0][:,:,2]
#     new_im[:images[0].shape[1],:images[0].shape[0],1] = images[0][:,:,1]
#     new_im[:images[0].shape[1],:images[0].shape[0],2] = images[0][:,:,0]
#     new_im[:images[1].shape[1],images[1].shape[0]:images[1].shape[0]*2,0] = images[1][:,:,2]
#     new_im[:images[1].shape[1],images[1].shape[0]:images[1].shape[0]*2,1] = images[1][:,:,1]
#     new_im[:images[1].shape[1],images[1].shape[0]:images[1].shape[0]*2,2] = images[1][:,:,0]
#     new_im[:images[2].shape[1],images[1].shape[0]*2:images[1].shape[0]*3,0] = images[2][:,:,2]
#     new_im[:images[2].shape[1],images[1].shape[0]*2:images[1].shape[0]*3,1] = images[2][:,:,1]
#     new_im[:images[2].shape[1],images[1].shape[0]*2:images[1].shape[0]*3,2] = images[2][:,:,0]
#     return new_im

def combine_horz(images):
    new_im = np.zeros((images[0].shape[0],images[0].shape[1]*2,3))
    new_im[:images[0].shape[1],:images[0].shape[0],0] = images[0][:,:,2]
    new_im[:images[0].shape[1],:images[0].shape[0],1] = images[0][:,:,1]
    new_im[:images[0].shape[1],:images[0].shape[0],2] = images[0][:,:,0]
    new_im[:images[1].shape[1],images[1].shape[0]:images[1].shape[0]*2,0] = images[1][:,:,2]
    new_im[:images[1].shape[1],images[1].shape[0]:images[1].shape[0]*2,1] = images[1][:,:,1]
    new_im[:images[1].shape[1],images[1].shape[0]:images[1].shape[0]*2,2] = images[1][:,:,0]
    return new_im

import matplotlib.cm
cmap =matplotlib.cm.get_cmap('jet')

outfolder = os.path.join('./visualizations/simvis/activation_maps/',datetime.now().strftime("%Y%m%d_%H%M%S"))
if not os.path.exists(outfolder):
    os.makedirs(outfolder)

for label in reppedLabels:
    sameInds = random.sample(indsByLabel[label],2)
    feat1 = trainingFeats[sameInds[0],:]
    feat2 = trainingFeats[sameInds[1],:]
    im1 = train_data.getBatchFromImageList([trainingIms[sameInds[0]]])
    squeezed_im1 = np.squeeze(im1)
    label1 = trainingLabels[sameInds[0]]
    im2 = train_data.getBatchFromImageList([trainingIms[sameInds[1]]])
    squeezed_im2 = np.squeeze(im2)
    label2 = trainingLabels[sameInds[1]]
    bestFeats = np.argsort(-(feat1*feat2))

    randInd = random.choice(indsByLabel[random.choice(reppedLabels[reppedLabels!=label1])])
    im3 = train_data.getBatchFromImageList([trainingIms[randInd]])
    squeezed_im3 = np.squeeze(im3)
    label3 = trainingLabels[randInd]

    batch[0,:,:,:] = im1
    batch[batch_size/2,:,:,:] = im2
    batch[batch_size-1,:,:,:] = im3

    labels[0] = label1
    labels[1] = label2
    labels[3] = label3

    ctr = 0
    g, wgts, cvout = sess.run([gap, weights, convOut],feed_dict={image_batch:batch, label_batch:labels,featInd:bestFeats[0]})
    for ft in bestFeats[:3]:
        wgt = wgts[:,ft]

        cvout1 = cvout[0,:,:,:].reshape((cvout.shape[1]*cvout.shape[2],cvout.shape[3])).transpose()
        cvout2 = cvout[batch_size/2,:,:,:].reshape((cvout.shape[1]*cvout.shape[2],cvout.shape[3])).transpose()
        cvout3 = cvout[batch_size-1,:,:,:].reshape((cvout.shape[1]*cvout.shape[2],cvout.shape[3])).transpose()

        bs,h,w,nc = cvout.shape

        cam1 = wgt.dot(cvout1).reshape(h,w)
        cam1 = cam1 - np.min(cam1)
        cam1 = cam1 / np.max(cam1)
        if feat1[ft] < 0:
            cam1 = 1 - cam1
        cam1 = zoom(cam1,float(crop_size[0])/float(cam1.shape[0]),order=1)
        hm1 = cmap(cam1)
        hm1 = hm1[:,:,:3]*255.

        bg1 = Image.fromarray(squeezed_im1.astype('uint8'))
        fg1 = Image.fromarray(hm1.astype('uint8'))
        im1_with_heatmap = np.array(Image.blend(bg1,fg1,alpha=.35).getdata()).reshape((crop_size[0],crop_size[1],3))

        cam2 = wgt.dot(cvout2).reshape(h,w)
        cam2 = cam2 - np.min(cam2)
        cam2 = cam2 / np.max(cam2)
        if feat2[ft] < 0:
            cam2 = 1 - cam2
        cam2 = zoom(cam2,float(crop_size[0])/float(cam2.shape[0]),order=1)
        hm2 = cmap(cam2)
        hm2 = hm2[:,:,:3]*255.

        bg2 = Image.fromarray(squeezed_im2.astype('uint8'))
        fg2 = Image.fromarray(hm2.astype('uint8'))
        im2_with_heatmap = np.array(Image.blend(bg2,fg2,alpha=.35).getdata()).reshape((crop_size[0],crop_size[1],3))

        # cam3 = wgt.dot(cvout3).reshape(h,w)
        # cam3 = cam3 - np.min(cam3)
        # cam3 = cam3 / np.max(cam3)
        # cam3 = zoom(cam3,float(crop_size[0])/float(cam3.shape[0]),order=1)
        # hm3 = cmap(cam3)
        # hm3 = hm3[:,:,:3]*255.
        #
        # bg3 = Image.fromarray(squeezed_im3.astype('uint8'))
        # fg3 = Image.fromarray(hm3.astype('uint8'))
        # im3_with_heatmap = np.array(Image.blend(bg3,fg3,alpha=.35).getdata()).reshape((crop_size[0],crop_size[1],3))

        # out_im = combine_horz([im1_with_heatmap,im2_with_heatmap,im3_with_heatmap])
        out_im = combine_horz([im1_with_heatmap,im2_with_heatmap])
        pil_out_im = Image.fromarray(out_im.astype('uint8'))
        feat_outfolder = os.path.join(outfolder,'%d'%(ft))
        if not os.path.exists(feat_outfolder):
            os.makedirs(feat_outfolder)
        pil_out_im.save(os.path.join(feat_outfolder,'%d_%.2f_%.2f.png'%(ctr,feat1[ft],feat2[ft])))
        ctr += 1
