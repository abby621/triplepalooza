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
from tensorflow.python.ops.image_ops_impl import *
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets import vgg

def main():
    ckpt_dir = './output/cifar/no_doctoring/ckpts'
    log_dir = './output/cifar/no_doctoring/logs'
    filename = './inputs/cifar/train.txt'
    mean_file = './models/cifar/cifar_mean_im.npy'
    pretrained_net = None
    img_size = [256, 256]
    crop_size = [224, 224]
    num_iters = 5000
    summary_iters = 10
    save_iters = 500
    learning_rate = .001
    margin = 10
    featLayer = 'vgg_16/fc7'

    batch_size = 30
    num_pos_examples = batch_size/10

    # Queuing op loads data into input tensor
    image_batch = tf.placeholder(tf.float32, shape=[batch_size, crop_size[0], crop_size[0], 3])
    people_mask_batch = tf.placeholder(tf.float32, shape=[batch_size, crop_size[0], crop_size[0], 1])
    label_batch = tf.placeholder(tf.int32, shape=(batch_size))

    # Create data "batcher"
    data = CombinatorialTripletSet(filename, mean_file, img_size, crop_size, batch_size, num_pos_examples)

    # after we've doctored everything, we need to remember to subtract off the mean
    repMeanIm = np.tile(np.expand_dims(data.meanImage,0),[batch_size,1,1,1])
    final_batch = tf.subtract(image_batch,repMeanIm)

    print("Preparing network...")
    with slim.arg_scope(vgg.vgg_arg_scope()):
            _, layers = vgg.vgg_16(final_batch,num_classes=100, is_training=True)

    feat = tf.squeeze(layers[featLayer])

    # expanded_a = tf.expand_dims(feat, 1)
    # expanded_b = tf.expand_dims(feat, 0)
    # D2 = tf.reduce_sum(tf.squared_difference(expanded_a, expanded_b), 2)

    expanded_a = tf.expand_dims(feat, 1)
    expanded_b = tf.expand_dims(feat, 0)
    D = tf.reduce_sum(tf.squared_difference(expanded_a, expanded_b), 2)

    D_max = tf.reduce_max(D)
    D_mean, D_var = tf.nn.moments(D, axes=[0,1])
    lowest_nonzero_distance = tf.reduce_max(-D)

    bottom_thresh = 1.2*lowest_nonzero_distance
    top_thresh = (D_max + D_mean)/2.0

    bool_mask = tf.logical_and(D>=bottom_thresh,D<=top_thresh)

    D2 = tf.multiply(D,tf.cast(bool_mask,tf.float32))

    posIdx = np.floor(np.arange(0,batch_size)/num_pos_examples).astype('int')
    posIdx10 = num_pos_examples*posIdx
    posImInds = np.tile(posIdx10,(num_pos_examples,1)).transpose()+np.tile(np.arange(0,num_pos_examples),(batch_size,1))
    anchorInds = np.tile(np.arange(0,batch_size),(num_pos_examples,1)).transpose()

    posImInds_flat = posImInds.ravel()
    anchorInds_flat = anchorInds.ravel()

    posPairInds = zip(posImInds_flat,anchorInds_flat)

    posDists = tf.reshape(tf.gather_nd(D2,posPairInds),(batch_size,num_pos_examples))

    shiftPosDists = tf.reshape(posDists,(1,batch_size,num_pos_examples))
    posDistsRep = tf.tile(shiftPosDists,(batch_size,1,1))

    allDists = tf.tile(tf.expand_dims(D2,2),(1,1,num_pos_examples))

    ra, rb, rc = np.meshgrid(np.arange(0,batch_size),np.arange(0,batch_size),np.arange(0,num_pos_examples))

    bad_negatives = np.floor((ra)/num_pos_examples) == np.floor((rb)/num_pos_examples)
    bad_positives = np.mod(rb,num_pos_examples) == np.mod(rc,num_pos_examples)

    mask = ((1-bad_negatives)*(1-bad_positives)).astype('float32')

    loss1 = tf.multiply(mask,margin + posDistsRep - allDists)
    loss2 = tf.maximum(0., loss1)
    loss3 = tf.reduce_mean(loss2)

    # slightly counterintuitive to not define "init_op" first, but tf vars aren't known until added to graph
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss3)
    summary_op = tf.summary.merge_all()
    init_op = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver(max_to_keep=20)

    # tf will consume any GPU it finds on the system. Following lines restrict it to "first" GPU
    c = tf.ConfigProto()
    c.gpu_options.visible_device_list="0,1"

    print("Starting session...")
    with tf.Session(config=c) as sess:
        sess.run(init_op)

        writer = tf.summary.FileWriter(log_dir, sess.graph)

        if pretrained_net:
            saver.restore(sess, pretrained_net)

        print("Start training...")
        ctr  = 0
        for step in range(num_iters):
            start_time = time.time()
            batch, labels, ims = data.getBatch()
            people_masks = data.getPeopleMasks()
            _, loss_val = sess.run([train_op, loss3], feed_dict={image_batch: batch, people_mask_batch: people_masks, label_batch: labels})
            # dd = sess.run(D, feed_dict={image_batch: batch, people_mask_batch: people_masks, label_batch: labels})
            # bb = sess.run(bool_mask, feed_dict={image_batch: batch, people_mask_batch: people_masks, label_batch: labels})
            dd2 = sess.run(D2, feed_dict={image_batch: batch, people_mask_batch: people_masks, label_batch: labels})
            # l1 = sess.run(loss1, feed_dict={image_batch: batch, people_mask_batch: people_masks, label_batch: labels})
            # print l1
            print len(np.where(dd2==0)[0])

            duration = time.time() - start_time

            # if step % summary_iters == 0:
            print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_val, duration))
            # Update the events file.
#                summary_str = sess.run(summary_op)
#                writer.add_summary(summary_str, step)
#                writer.flush()
#
            # Save a checkpoint
            if (step + 1) % save_iters == 0 or (step + 1) == num_iters:
                print('Saving checkpoint at iteration: %d' % (step))
                pretrained_net = os.path.join(ckpt_dir, 'checkpoint')
                saver.save(sess, pretrained_net, global_step=step)

      #  coord.request_stop()
       # coord.join(threads)

if __name__ == "__main__":
    main()