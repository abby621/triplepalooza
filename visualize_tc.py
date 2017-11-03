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

train_file = './inputs/traffickcam/train_equal.txt'
test_file = './inputs/traffickcam/test_equal.txt'
pretrained_net = './output/traffickcam/ckpts/checkpoint-201711011620_lr0pt0001_outputSz128_margin0pt3-22293'
img_size = [256, 256]
crop_size = [224, 224]
featLayer = 'resnet_v2_50/logits'
convOutLayer = 'resnet_v2_50/block4'
mean_file = './models/mars/mean_im.npy'

batch_size = 100
num_pos_examples = batch_size/10

output_size = 128

# Create test_data "batcher"
train_data = CombinatorialTripletSet(train_file, mean_file, img_size, crop_size, batch_size, num_pos_examples,is_training=False)
test_data = CombinatorialTripletSet(test_file, mean_file, img_size, crop_size, batch_size, num_pos_examples,is_training=False)

image_batch = tf.placeholder(tf.float32, shape=[batch_size, crop_size[0], crop_size[0], 3])
repMeanIm = np.tile(np.expand_dims(test_data.meanImage,0),[batch_size,1,1,1])
final_batch = tf.subtract(image_batch,repMeanIm)
label_batch = tf.placeholder(tf.int32, shape=(batch_size))

print("Preparing network...")
with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    _, layers = resnet_v2.resnet_v2_50(final_batch, num_classes=output_size, is_training=True)

feat = tf.nn.l2_normalize(layers[featLayer],3)
# feat = tf.nn.l2_normalize(tf.get_default_graph().get_tensor_by_name("resnet_v2_50/pool5:0"),3)
convOut = tf.nn.l2_normalize(layers[convOutLayer],3)
weights = tf.squeeze(tf.get_default_graph().get_tensor_by_name("resnet_v2_50/logits/weights:0"))
gap = tf.squeeze(tf.nn.l2_normalize(tf.get_default_graph().get_tensor_by_name("resnet_v2_50/pool5:0"),3))

featInd = tf.placeholder(tf.int32, shape=())
saliency_map = tf.gradients(tf.gather(tf.squeeze(feat),featInd), final_batch)[0]

filters = tf.gradients(tf.gather(tf.squeeze(convOut),featInd), final_batch)[0]

c = tf.ConfigProto()
c.gpu_options.visible_device_list="0,1"
sess = tf.Session(config=c)
saver = tf.train.Saver(max_to_keep=100)
saver.restore(sess, pretrained_net)

logs_path = '/tmp/tensorflow/'

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(logs_path,sess.graph)

testingImsAndLabels = [(test_data.files[ix][iy],test_data.classes[ix]) for ix in range(len(test_data.files)) for iy in range(len(test_data.files[ix]))]
prunedTestingImsAndLabels = []
prunedTestingIms = []
for im, label in testingImsAndLabels:
    im_end = im.split('/')[-1]
    if im_end not in prunedTestingIms:
        prunedTestingIms.append(im_end)
        prunedTestingImsAndLabels.append([im,label])

testingImsAndLabels = prunedTestingImsAndLabels

# random.shuffle(testingImsAndLabels)
testingImsAndLabels = testingImsAndLabels[:5000]
random.shuffle(testingImsAndLabels)
numTestingIms = len(testingImsAndLabels)

testingLabels = np.array([label for (im,label) in testingImsAndLabels])
indsByLabel = {}
for label in np.unique(testingLabels):
    goodInds = np.where(testingLabels==label)[0]
    if len(goodInds) > 1:
        indsByLabel[label] = goodInds

reppedLabels = np.array(indsByLabel.keys())

testingFeats = np.empty((numTestingIms,gap.shape[1]),dtype=np.float32)
testingIms = np.empty((numTestingIms),dtype=object)
testingLabels = np.empty((numTestingIms),dtype=np.int32)
for idx in range(0,numTestingIms,batch_size):
    print idx, '/', numTestingIms
    il = testingImsAndLabels[idx:idx+batch_size]
    ims = [i[0] for i in il]
    labels = [i[1] for i in il]
    batch = test_data.getBatchFromImageList(ims)
    testingIms[idx:idx+batch_size] = ims
    testingLabels[idx:idx+batch_size] = labels
    ff = sess.run(gap, feed_dict={image_batch: batch, label_batch:labels})
    testingFeats[idx:idx+batch_size,:] = np.squeeze(ff)

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
    new_im = np.zeros((images[0].shape[0],images[0].shape[1]*len(images),3))
    for idx in range(len(images)):
        new_im[:images[idx].shape[1],images[idx].shape[0]*idx:images[idx].shape[0]*idx+images[idx].shape[0],0] = images[idx][:,:,2]
        new_im[:images[idx].shape[1],images[idx].shape[0]*idx:images[idx].shape[0]*idx+images[idx].shape[0],1] = images[idx][:,:,1]
        new_im[:images[idx].shape[1],images[idx].shape[0]*idx:images[idx].shape[0]*idx+images[idx].shape[0],2] = images[idx][:,:,0]

    return new_im

def combine_vert(images):
    new_im = np.zeros((images[0].shape[0]*len(images),images[0].shape[1],3))
    for idx in range(len(images)):
        new_im[images[idx].shape[0]*idx:images[idx].shape[0]*idx+images[idx].shape[0],:images[idx].shape[1],:] = images[idx]

    return new_im

import matplotlib.cm
cmap =matplotlib.cm.get_cmap('jet')

outfolder = os.path.join('./visualizations/simvis/traffickcam/',datetime.now().strftime("%Y%m%d_%H%M%S"))
if not os.path.exists(outfolder):
    os.makedirs(outfolder)
    os.makedirs(os.path.join(outfolder,'by_hotel'))
    # os.makedirs(os.path.join(outfolder,'by_feature'))

def getHeatMap(mask,im):
    # cam = np.sum(imIn,axis=0)
    cam = mask - np.min(mask)
    cam = cam / np.max(cam)
    cam = zoom(cam,float(crop_size[0])/float(cam.shape[0]),order=1)
    hm = cmap(cam)
    hm = hm[:,:,:3]*255.
    bg = Image.fromarray(im.astype('uint8'))
    fg = Image.fromarray(hm.astype('uint8'))
    im_with_heatmap = np.array(Image.blend(bg,fg,alpha=.35).getdata()).reshape((crop_size[0],crop_size[1],3))
    return im_with_heatmap

def getDist(feat,otherFeats):
    dist = (otherFeats - feat)**2
    dist = np.sum(dist,axis=1)
    dist = np.sqrt(dist)
    # dist = np.array([np.dot(feat,otherFeat) for otherFeat in otherFeats])
    return dist

for label in reppedLabels:
    for idx in np.where(testingLabels==label)[0]:
        thisIm = testingIms[idx]
        thisFeat = testingFeats[idx,:]
        dists = getDist(thisFeat,testingFeats)
        sortedInds = np.argsort(dists)[1:]
        sortedLabels = testingLabels[sortedInds][1:]
        sortedIms = testingIms[sortedInds][1:]
        topHit = np.where(sortedLabels==label)[0][0]

        feat0 = testingFeats[idx,:]
        im0 = test_data.getBatchFromImageList([thisIm])
        squeezed_im0 = np.squeeze(im0)
        label0 = testingLabels[idx]

        # top result
        feat1 = testingFeats[sortedInds[0],:]
        im1 = test_data.getBatchFromImageList([testingIms[sortedInds[0]]])
        squeezed_im1 = np.squeeze(im1)
        label1 = testingLabels[sortedInds[0]]
        featDists1 = (feat0*feat1)
        sortedDists1 = np.sort(featDists1)[::-1]
        sumTo1 = [np.sum(sortedDists1[:aa]) for aa in range(len(sortedDists1))]
        # cutOffInd1 = np.where(sumTo1>sumTo1[-1]*.5)[0][0]
        bestFeats1 = np.argsort(-featDists1)

        # top correct match
        feat2 = testingFeats[sortedInds[topHit],:]
        im2 = test_data.getBatchFromImageList([testingIms[sortedInds[topHit]]])
        squeezed_im2 = np.squeeze(im2)
        label2 = testingLabels[sortedInds[topHit]]
        featDists2 = (feat0*feat2)
        sortedDists2 = np.sort(featDists2)[::-1]
        sumTo2 = [np.sum(sortedDists2[:aa]) for aa in range(len(sortedDists2))]
        # cutOffInd2 = np.where(sumTo2>sumTo2[-1]*.5)[0][0]
        bestFeats2 = np.argsort(-featDists2)

        # interleave the results in the batch
        batch[0,:,:,:] = im0
        batch[batch_size/4,:,:,:] = im1
        batch[batch_size/4*3,:,:,:] = im2

        labels[0] = label0
        labels[batch_size/4] = label1
        labels[batch_size/4*3] = label2

        g, wgts, cvout = sess.run([gap, weights, convOut],feed_dict={image_batch:batch, label_batch:labels, featInd:bestFeats1[0]})
        bs,h,w,nc = cvout.shape

        # get heat maps
        # top match
        hm1_1 = getHeatMap(cvout[0,:,:,bestFeats1[0]],squeezed_im0)
        hm1_2 = getHeatMap(cvout[0,:,:,bestFeats1[1]],squeezed_im0)
        hm1_3 = getHeatMap(cvout[0,:,:,bestFeats1[2]],squeezed_im0)

        hm2_1 = getHeatMap(cvout[batch_size/4,:,:,bestFeats1[0]],squeezed_im1)
        hm2_2 = getHeatMap(cvout[batch_size/4,:,:,bestFeats1[1]],squeezed_im1)
        hm2_3 = getHeatMap(cvout[batch_size/4,:,:,bestFeats1[2]],squeezed_im1)

        # top correct match
        hm3_1 = getHeatMap(cvout[batch_size/4*3,:,:,bestFeats2[0]],squeezed_im2)
        hm3_2 = getHeatMap(cvout[batch_size/4*3,:,:,bestFeats2[1]],squeezed_im2)
        hm3_3 = getHeatMap(cvout[batch_size/4*3,:,:,bestFeats2[2]],squeezed_im2)

        out_im1 = combine_horz([hm1_1,hm1_2,hm1_3])
        out_im2 = combine_horz([hm2_1,hm2_2,hm2_3])
        out_im3 = combine_horz([hm3_1,hm3_2,hm3_3])

        # top row = top match; center = query; bottom = top correct match
        out_im4 = combine_vert([out_im2,out_im1,out_im3])
        pil_out_im = Image.fromarray(out_im4.astype('uint8'))

        hotel_outfolder = os.path.join(outfolder,'by_hotel',str(label))
        if not os.path.exists(hotel_outfolder):
            os.makedirs(hotel_outfolder)

        pil_out_im.save(os.path.join(hotel_outfolder,'%d_%d_%.3f_%.3f_%.3f_%.3f.png'%(idx,topHit,sumTo1[-1],sumTo1[2],sumTo2[-1],sumTo2[2])))
        print idx

        # feat_outfolder = os.path.join(outfolder,'by_feature',str(ft))
        # if not os.path.exists(feat_outfolder):
        #     os.makedirs(feat_outfolder)
        #
        # pil_out_im.save(os.path.join(feat_outfolder,'%d_%.2d.png'%(idx,cutOffInd)))
        # print ctr
        # ctr += 1

        # for ft in bestFeats[:3]:
        #     # wgt = wgts[:,ft]
        #     cvout1 = cvout[0,:,:,:].reshape((cvout.shape[1]*cvout.shape[2],cvout.shape[3])).transpose()
        #     cvout2 = cvout[batch_size/4,:,:,:].reshape((cvout.shape[1]*cvout.shape[2],cvout.shape[3])).transpose()
        #     cvout3 = cvout[batch_size/4*2,:,:,:].reshape((cvout.shape[1]*cvout.shape[2],cvout.shape[3])).transpose()
        #     cvout4 = cvout[batch_size/4*3,:,:,:].reshape((cvout.shape[1]*cvout.shape[2],cvout.shape[3])).transpose()
        #
        #     bs,h,w,nc = cvout.shape
        #
        #     # cam1 = wgt.dot(cvout1).reshape(h,w)
        #     cam1 = cvout1[ft,:].reshape(h,w)
        #     cam1 = cam1 - np.min(cam1)
        #     cam1 = cam1 / np.max(cam1)
        #     if feat1[ft] < 0:
        #         cam1 = 1 - cam1
        #
        #     cam1 = zoom(cam1,float(crop_size[0])/float(cam1.shape[0]),order=1)
        #     hm1 = cmap(cam1)
        #     hm1 = hm1[:,:,:3]*255.
        #
        #     bg1 = Image.fromarray(squeezed_im1.astype('uint8'))
        #     fg1 = Image.fromarray(hm1.astype('uint8'))
        #     im1_with_heatmap = np.array(Image.blend(bg1,fg1,alpha=.35).getdata()).reshape((crop_size[0],crop_size[1],3))
        #
        #     # cam2 = wgt.dot(cvout2).reshape(h,w)
        #     cam2 = cvout2[ft,:].reshape(h,w)
        #     cam2 = cam2 - np.min(cam2)
        #     cam2 = cam2 / np.max(cam2)
        #     if feat2[ft] < 0:
        #         cam2 = 1 - cam2
        #
        #     cam2 = zoom(cam2,float(crop_size[0])/float(cam2.shape[0]),order=1)
        #     hm2 = cmap(cam2)
        #     hm2 = hm2[:,:,:3]*255.
        #
        #     bg2 = Image.fromarray(squeezed_im2.astype('uint8'))
        #     fg2 = Image.fromarray(hm2.astype('uint8'))
        #     im2_with_heatmap = np.array(Image.blend(bg2,fg2,alpha=.35).getdata()).reshape((crop_size[0],crop_size[1],3))
        #
        #     # cam3 = wgt.dot(cvout3).reshape(h,w)
        #     cam3 = cvout3[ft,:].reshape(h,w)
        #     cam3 = cam3 - np.min(cam3)
        #     cam3 = cam3 / np.max(cam3)
        #     if feat3[ft] < 0:
        #         cam3 = 1 - cam3
        #
        #     cam3 = zoom(cam3,float(crop_size[0])/float(cam3.shape[0]),order=1)
        #     hm3 = cmap(cam3)
        #     hm3 = hm3[:,:,:3]*255.
        #
        #     bg3 = Image.fromarray(squeezed_im3.astype('uint8'))
        #     fg3 = Image.fromarray(hm3.astype('uint8'))
        #     im3_with_heatmap = np.array(Image.blend(bg3,fg3,alpha=.35).getdata()).reshape((crop_size[0],crop_size[1],3))
        #
        #     # cam4 = wgt.dot(cvout4).reshape(h,w)
        #     cam4 = cvout4[ft,:].reshape(h,w)
        #     cam4 = cam4 - np.min(cam4)
        #     cam4 = cam4 / np.max(cam4)
        #     if feat3[ft] < 0:
        #         cam4 = 1 - cam4
        #
        #     cam4 = zoom(cam4,float(crop_size[0])/float(cam4.shape[0]),order=1)
        #     hm4 = cmap(cam4)
        #     hm4 = hm4[:,:,:3]*255.
        #
        #     bg4 = Image.fromarray(squeezed_im4.astype('uint8'))
        #     fg4 = Image.fromarray(hm4.astype('uint8'))
        #     im4_with_heatmap = np.array(Image.blend(bg4,fg4,alpha=.35).getdata()).reshape((crop_size[0],crop_size[1],3))
        #
        #     out_im = combine_horz([im1_with_heatmap,im2_with_heatmap,im3_with_heatmap,im4_with_heatmap])
        #     pil_out_im = Image.fromarray(out_im.astype('uint8'))
        #
        #     person_outfolder = os.path.join(outfolder,'by_person',str(label))
        #     if not os.path.exists(person_outfolder):
        #         os.makedirs(person_outfolder)
        #
        #     pil_out_im.save(os.path.join(person_outfolder,'%d_%d_%.2f_%.2f_%.2f_%.2f.png'%(ctr,ft,feat1[ft],feat2[ft],feat3[ft],feat4[ft])))
        #
        #     feat_outfolder = os.path.join(outfolder,'by_feature',str(ft))
        #     if not os.path.exists(feat_outfolder):
        #         os.makedirs(feat_outfolder)
        #
        #     pil_out_im.save(os.path.join(feat_outfolder,'%d_%.2f_%.2f_%.2f_%.2f.png'%(ctr,feat1[ft],feat2[ft],feat3[ft],feat4[ft])))
        #     print ctr
        #     ctr += 1
