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

def main():
    filename = './inputs/mnist/test.txt'
    checkpoint_file = './output/ckpts/checkpoint-999'
    img_size = [28, 28]
    crop_size = [24, 24]
    featLayer = 'prob'

    batch_size = 100
    num_pos_examples = 10

    image_batch = tf.placeholder(tf.float32, shape=[batch_size, crop_size[0], crop_size[0], 3])
    label_batch = tf.placeholder(tf.int32, shape=(batch_size))

    print("Preparing network...")
    net = CaffeNetPlaces365({'data': image_batch})
    feat = net.layers[featLayer]

    # Create a saver for writing loading checkpoints.
    saver = tf.train.Saver()

    # Create data "batcher"
    data = CombinatorialTripletSet(filename, img_size, crop_size, batch_size, num_pos_examples)

    print("Starting session...")
    with tf.Session() as sess:
        # Here's where we need to load saved weights
        saver.restore(sess, checkpoint_file)

        allFeats = []
        num_iters = len(data.files) / 2 # Total hack: Just make sure to have even # of lines of test examples

        for step in range(num_iters):
            start_time = time.time()
            batch, np = data.getBatch()

            f = sess.run(feat, feed_dict={image_batch: batch})
            for aa in f:
                print np.where(np.array(aa)==np.max(aa))

            allFeats.extend(f)
            duration = time.time() - start_time

            # Write the summaries and print an overview
            if step % 1000 == 0:
                print('Step %d: (%.3f sec)' % (step, duration))

if __name__ == "__main__":
    main()
