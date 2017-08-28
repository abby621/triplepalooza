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

def main():
    ckpt_dir = './output/ckpts'
    log_dir = './output/logs'
    filename = './inputs/mnist/train.txt'
    img_size = [28, 28]
    crop_size = [24, 24]
    num_iters = 1000
    summary_iters = 1
    save_iters = 100
    learning_rate = .01
    margin = 10
    featLayer = 'fc7'

    batch_size = 100
    num_pos_examples = 10

    # Queuing op loads data into input tensor
    image_batch = tf.placeholder(tf.float32, shape=[batch_size, crop_size[0], crop_size[0], 3])
    label_batch = tf.placeholder(tf.int32, shape=(batch_size))

    print("Preparing network...")
    net = CaffeNetPlaces365({'data': image_batch})
    feat = net.layers[featLayer]

    r = tf.reduce_sum(feat*feat,1)
    r = tf.reshape(r,[-1, 1])
    D = r - 2*tf.matmul(feat, tf.transpose(feat)) + tf.transpose(r)

    posIdx = np.floor(np.arange(0,batch_size)/num_pos_examples).astype('int')
    posIdx10 = num_pos_examples*posIdx
    posImInds = np.tile(posIdx10,(num_pos_examples,1)).transpose()+np.tile(np.arange(0,num_pos_examples),(batch_size,1))
    anchorInds = np.tile(np.arange(0,batch_size),(num_pos_examples,1)).transpose()

    posImInds_flat = posImInds.ravel()
    anchorInds_flat = anchorInds.ravel()

    posPairInds = zip(posImInds_flat,anchorInds_flat)

    posDists = tf.reshape(tf.gather_nd(D,posPairInds),(batch_size,num_pos_examples))
    shiftPosDists = tf.reshape(posDists,(1,batch_size,num_pos_examples))
    posDistsRep = tf.tile(shiftPosDists,(batch_size,1,1))

    allDists = tf.tile(tf.expand_dims(D,2),(1,1,num_pos_examples))

    ra, rb, rc = np.meshgrid(np.arange(0,100),np.arange(0,100),np.arange(0,10))

    bad_negatives = np.floor((ra)/10) == np.floor((rb)/10)
    bad_positives = np.mod(rb,10) == np.mod(rc,10)

    mask = ((1-bad_negatives)*(1-bad_positives)).astype('float32')

    posDistsFinal = tf.multiply(mask,posDistsRep)
    allDistsFinal = tf.multiply(mask,allDists)

    loss = tf.maximum(0., margin + posDistsFinal - allDistsFinal)
    loss = tf.reduce_mean(loss)

    # slightly counterintuitive to not define "init_op" first, but tf vars aren't known until added to graph
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    summary_op = tf.summary.merge_all()
    init_op = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver(max_to_keep=20)

    # Create data "batcher"
    data = CombinatorialTripletSet(filename, img_size, crop_size, batch_size, num_pos_examples)

    print("Starting session...")
    with tf.Session() as sess:
        sess.run(init_op)

        writer = tf.summary.FileWriter(log_dir, sess.graph)

        # net.load('./models/mnist/mnist_model.npy', sess)

        print("Start training...")
        for step in range(num_iters):
            start_time = time.time()
            batch, labels = data.getBatch()
            _, loss_val = sess.run([train_op, loss], feed_dict={image_batch: batch, label_batch: labels})
            duration = time.time() - start_time

            # Write the summaries and print an overview
            if step % summary_iters == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_val, duration))
                # Update the events file.
#                summary_str = sess.run(summary_op)
#                writer.add_summary(summary_str, step)
#                writer.flush()
#
            # Save a checkpoint
            if (step + 1) % save_iters == 0 or (step + 1) == num_iters:
                print('Saving checkpoint at iteration: %d' % (step))
                checkpoint_file = os.path.join(ckpt_dir, 'checkpoint')
                saver.save(sess, checkpoint_file, global_step=step)

      #  coord.request_stop()
       # coord.join(threads)

if __name__ == "__main__":
    main()
