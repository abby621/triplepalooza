# -*- coding: utf-8 -*-
"""
# python cifar_training_no_doctoring.py margin output_size learning_rate is_overfitting
# If overfitting:
# python cifar_training_no_doctoring.py 5 12 .00005 True
# Else:
# python cifar_training_no_doctoring.py 5 12 .00005 False
"""

import tensorflow as tf
from classfile import CombinatorialTripletSet
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
from tensorflow.contrib.slim.python.slim.nets import resnet_v1

import signal
import time
import sys

def main(margin,output_size,learning_rate,is_overfitting):
    def handler(signum, frame):
        print 'Saving checkpoint before closing'
        pretrained_net = os.path.join(ckpt_dir, 'checkpoint')
        saver.save(sess, pretrained_net, global_step=step)
        print 'Checkpoint-',step, ' saved!'
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)

    ckpt_dir = './output/cifar/no_doctoring/ckpts'
    log_dir = './output/cifar/no_doctoring/logs'
    train_filename = './inputs/cifar/train.txt'
    mean_file = './models/cifar/cifar_mean_im.npy'
    pretrained_net = None
    img_size = [256, 256]
    crop_size = [224, 224]
    num_iters = 20000
    summary_iters = 10
    save_iters = 1000
    featLayer = 'resnet_v1_50/logits'
    is_training = True

    margin = int(margin)
    output_size = int(output_size)
    learning_rate = float(learning_rate)
    is_overfitting = True

    batch_size = 100
    num_pos_examples = batch_size/10

    # Create data "batcher"
    train_data = CombinatorialTripletSet(train_filename, mean_file, img_size, crop_size, batch_size, num_pos_examples, isTraining=is_training, isOverfitting=is_overfitting)
    numClasses = len(train_data.files)
    numIms = np.sum([len(train_data.files[idx]) for idx in range(0,numClasses)])
    datestr = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_log_file = open(os.path.join(log_dir,datestr)+'_lr'+str(learning_rate)+'_outputSz'+str(output_size)+'_margin'+str(margin)+'_train.txt','a')
    print '------------'
    print ''
    print 'Going to train with the following parameters:'
    print '# Classes: ',numClasses
    print '# Ims: ',numIms
    print 'Margin: ',margin
    print 'Output size: ', output_size
    print 'Learning rate: ',learning_rate
    print 'Overfitting?: ',is_overfitting
    print ''
    print '------------'

    # Queuing op loads data into input tensor
    image_batch = tf.placeholder(tf.float32, shape=[batch_size, crop_size[0], crop_size[0], 3])
    people_mask_batch = tf.placeholder(tf.float32, shape=[batch_size, crop_size[0], crop_size[0], 1])
    label_batch = tf.placeholder(tf.int32, shape=(batch_size))

    # after we've doctored everything, we need to remember to subtract off the mean
    repMeanIm = np.tile(np.expand_dims(train_data.meanImage,0),[batch_size,1,1,1])
    noise = tf.random_normal(shape=[batch_size, crop_size[0], crop_size[0], 3], mean=0.0, stddev=3, dtype=tf.float32)
    final_batch = tf.add(tf.subtract(image_batch,repMeanIm),noise)

    print("Preparing network...")
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        _, layers = resnet_v1.resnet_v1_50(final_batch, num_classes=output_size, is_training=True)

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
    sess = tf.Session(config=c)
    sess.run(init_op)

    writer = tf.summary.FileWriter(log_dir, sess.graph)

    if pretrained_net:
        saver.restore(sess, pretrained_net)

    print("Start training...")
    ctr  = 0
    for step in range(num_iters):
        start_time1 = time.time()
        batch, labels, ims = train_data.getBatch()
        _, loss_val = sess.run([train_op, loss3], feed_dict={image_batch: batch, label_batch: labels})
        end_time2 = time.time()
        duration = end_time2-start_time1
        out_str = 'Step %d: loss = %.2f (%.3f sec)' % (step, loss_val, duration)
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

      #  coord.request_stop()
       # coord.join(threads)

if __name__ == "__main__":
    args = sys.argv
    if len(args) < 5:
        print 'Expected four input parameters: margin, output_size, learning_rate, is_overfitting'
    margin = args[1]
    output_size = args[2]
    learning_rate = args[3]
    is_overfitting = args[4]
    main(margin,output_size,learning_rate,is_overfitting)
