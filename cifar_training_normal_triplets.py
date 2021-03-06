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
    ckpt_dir = './output/cifar/normal_triplets/ckpts'
    log_dir = './output/cifar/normal_triplets/logs'
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

    # Queuing op loads data into input tensor
    image_batch = tf.placeholder(tf.float32, shape=[batch_size, crop_size[0], crop_size[0], 3])
    label_batch = tf.placeholder(tf.int32, shape=(batch_size))

    # Create data "batcher"
    data = VanillaTripletSet(filename, mean_file, img_size, crop_size, batch_size)

    # after we've doctored everything, we need to remember to subtract off the mean
    repMeanIm = np.tile(np.expand_dims(data.meanImage,0),[batch_size,1,1,1])
    final_batch = tf.subtract(image_batch,repMeanIm)

    print("Preparing network...")
    with slim.arg_scope(vgg.vgg_arg_scope()):
            _, layers = vgg.vgg_16(final_batch, num_classes=100, is_training=True)

    feat = tf.squeeze(layers[featLayer])

    anchorFeats = feat[::3]
    posFeats = feat[1::3]
    negFeats = feat[2::3]

    posDists = tf.abs(anchorFeats-posFeats)
    negDists = tf.abs(anchorFeats-negFeats)

    loss = margin + posDists - negDists
    loss = tf.maximum(0., loss)
    loss = tf.reduce_mean(loss)

    # slightly counterintuitive to not define "init_op" first, but tf vars aren't known until added to graph
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    summary_op = tf.summary.merge_all()
    init_op = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver(max_to_keep=20)

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
            _, loss_val = sess.run([train_op, loss], feed_dict={image_batch: batch, label_batch: labels})
            # dd = sess.run(D, feed_dict={image_batch: batch, people_mask_batch: people_masks, label_batch: labels})
            # dd2 = sess.run(D2, feed_dict={image_batch: batch, people_mask_batch: people_masks, label_batch: labels})
            # print len(np.where(dd==0)[0]),len(np.where(dd2==0)[0])

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
