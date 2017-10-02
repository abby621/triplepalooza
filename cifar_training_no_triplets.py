# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 13:02:07 2016

@author: souvenir
"""

import tensorflow as tf
from classfile import NonTripletSet
import os.path
import time
from datetime import datetime
import numpy as np
from PIL import Image
from tensorflow.python.ops.image_ops_impl import *
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets import resnet_v2

def main():
    ckpt_dir = './output/cifar/no_triplets/ckpts'
    log_dir = './output/cifar/no_triplets/logs'
    train_log_file = open(os.path.join(log_dir,str(datetime.now())+'_train.txt'),'a')
    test_log_file = open(os.path.join(log_dir,str(datetime.now())+'_test.txt'),'a')
    train_filename = './inputs/cifar/train.txt'
    test_filename = './inputs/cifar/test.txt'
    mean_file = './models/cifar/cifar_mean_im.npy'
    pretrained_net = None
    img_size = [256, 256]
    crop_size = [224, 224]
    num_iters = 200000
    summary_iters = 10
    save_iters = 1000
    learning_rate = .0001
    margin = 10
    featLayer = 'resnet_v2_50/logits'

    batch_size = 90
    num_pos_examples = batch_size/10

    # Queuing op loads train_data into input tensor
    image_batch = tf.placeholder(tf.float32, shape=[batch_size, crop_size[0], crop_size[0], 3])
    label_batch = tf.placeholder(tf.int32, shape=(batch_size))

    # Create train_data "batcher"
    train_data = NonTripletSet(train_filename, mean_file, img_size, crop_size, batch_size, num_pos_examples)
    test_data = NonTripletSet(test_filename, mean_file, img_size, crop_size, batch_size, num_pos_examples)

    # after we've doctored everything, we need to remember to subtract off the mean
    repMeanIm = np.tile(np.expand_dims(train_data.meanImage,0),[batch_size,1,1,1])
    noise = tf.random_normal(shape=[batch_size, crop_size[0], crop_size[0], 3], mean=0.0, stddev=3, dtype=tf.float32)
    final_batch = tf.add(tf.subtract(image_batch,repMeanIm),noise)

    print("Preparing network...")
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        _, layers = resnet_v2.resnet_v2_50(final_batch, num_classes=100, is_training=True)

    feat = tf.squeeze(layers[featLayer])
    prediction=tf.argmax(feat,1)
    aa = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=feat, labels=label_batch)
    loss = tf.reduce_mean(aa)

    # slightly counterintuitive to not define "init_op" first, but tf vars aren't known until added to graph
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    summary_op = tf.summary.merge_all()
    init_op = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver(max_to_keep=100)

    c = tf.ConfigProto()
    c.gpu_options.visible_device_list="0,1"

    print("Starting session...")
    sess = tf.Session(config=c)
    sess.run(init_op)

    if pretrained_net:
        saver.restore(sess, pretrained_net)

    writer = tf.summary.FileWriter(log_dir, sess.graph)

    print("Start training...")
    ctr  = 0
    for step in range(num_iters):
        start_time1 = time.time()
        batch, labels, ims = train_data.getBatch()
        _, loss_val, pred = sess.run([train_op, loss, prediction], feed_dict={image_batch: batch, label_batch: labels})
        train_accuracy = int(100*float(len(np.where(pred==labels)[0]))/float(batch_size))
        end_time2 = time.time()
        duration = end_time2-start_time1
        if step % 50 == 0:
            for ix in range(0,10):
                test_batch, test_labels, test_ims = test_data.getBatch()
                test_best = sess.run([prediction],feed_dict={image_batch:test_batch,label_batch:test_labels})
                test_accuracy = int(100*float(len(np.where(test_best==test_labels)[0]))/float(batch_size))
                # if step % summary_iters == 0:
                out_str = 'TEST: Step %d: top1-accuracy: %d' % (step,test_accuracy)
                print(out_str)
                test_log_file.write(out_str+'\n')
        out_str = 'Step %d: loss = %.2f (%.3f sec), top1-accuracy: %d' % (step, loss_val, duration,train_accuracy)
        print(out_str)
        train_log_file.write(out_str+'\n')
        # Update the events file.
        # summary_str = sess.run(summary_op)
        # writer.add_summary(summary_str, step)
        # writer.flush()
        #
        # Save a checkpoint
        if (step + 1) % save_iters == 0 or (step + 1) == num_iters:
            print('Saving checkpoint at iteration: %d' % (step))
            pretrained_net = os.path.join(ckpt_dir, 'checkpoint')
            saver.save(sess, pretrained_net, global_step=step)

    sess.close()
    train_log_file.close()
    test_log_file.close()

if __name__ == "__main__":
    main()
