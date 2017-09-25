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
from PIL import Image
from tensorflow.python.ops.image_ops_impl import *
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

def main():
    ckpt_dir = './output/ckpts'
    log_dir = './output/logs'
    filename = './inputs/traffickcam/train_equal.txt'
    mean_file = './models/places365/places365CNN_mean.npy'
    pretrained_net = './models/places365/alexnet.npy'
    img_size = [256, 256]
    crop_size = [227, 227]
    num_iters = 100000
    summary_iters = 10
    save_iters = 500
    learning_rate = .0001
    margin = 10
    featLayer = 'fc7'

    batch_size = 100
    num_pos_examples = batch_size/10

    # Queuing op loads data into input tensor
    image_batch = tf.placeholder(tf.float32, shape=[batch_size, crop_size[0], crop_size[0], 3])
    people_mask_batch = tf.placeholder(tf.float32, shape=[batch_size, crop_size[0], crop_size[0], 1])
    label_batch = tf.placeholder(tf.int32, shape=(batch_size))

    # Create data "batcher"
    data = CombinatorialTripletSet(filename, mean_file, img_size, crop_size, batch_size, num_pos_examples)

    # doctor image params
    percent_crop = .5
    percent_people = .5
    percent_rotate = .2
    percent_filters = .4
    percent_text = .1

    # richard's argument: since the data is randomly loaded, we don't need to change the indices that we perform operations on every time; i am on board with this, but had already implemented the random crops, so will leave that for now
    # apply random rotations
    num_rotate = int(batch_size*percent_rotate)
    rotate_inds = np.random.choice(np.arange(0,batch_size),num_rotate,replace=False)
    rotate_vals = np.random.randint(-65,65,num_rotate).astype('float32')/float(100)
    rotate_angles = np.zeros((batch_size))
    rotate_angles[rotate_inds] = rotate_vals
    rotated_batch = tf.contrib.image.rotate(image_batch,rotate_angles,interpolation='BILINEAR')

    # do random crops
    num_to_crop = int(batch_size*percent_crop)
    num_to_not_crop = batch_size - num_to_crop
    shuffled_inds = tf.random_shuffle(np.arange(0,batch_size,dtype='int32'))
    crop_inds = tf.slice(shuffled_inds,[0],[num_to_crop])
    uncropped_inds = tf.slice(shuffled_inds,[num_to_crop],[num_to_not_crop])

    crop_ratio = float(3)/float(5)
    crop_yx = tf.random_uniform([num_to_crop,2], 0,1-crop_ratio, dtype=tf.float32, seed=0)
    crop_sz = tf.add(crop_yx,np.tile([crop_ratio,crop_ratio],[num_to_crop, 1]))
    crop_boxes = tf.concat([crop_yx,crop_sz],axis=1)

    uncropped_boxes = np.tile([0,0,1,1],[num_to_not_crop,1])

    all_inds = tf.concat([crop_inds,uncropped_inds],axis=0)
    all_boxes = tf.concat([crop_boxes,uncropped_boxes],axis=0)

    cropped_batch = tf.image.crop_and_resize(rotated_batch,all_boxes,all_inds,crop_size)

    # insert people masks
    num_people_masks = int(batch_size*percent_people)
    mask_inds = np.random.choice(np.arange(0,batch_size),num_people_masks,replace=False)

    start_masks = np.zeros([batch_size, crop_size[0], crop_size[0], 1],dtype='float32')
    start_masks[mask_inds,:,:,:] = 1

    inv_start_masks = np.ones([batch_size, crop_size[0], crop_size[0], 1],dtype='float32')
    inv_start_masks[mask_inds,:,:,:] = 0

    masked_masks = tf.add(inv_start_masks,tf.cast(tf.multiply(people_mask_batch,start_masks),dtype=tf.float32))
    masked_masks2 = tf.cast(tf.tile(masked_masks,[1, 1, 1, 3]),dtype=tf.float32)
    masked_batch = tf.multiply(masked_masks,cropped_batch)

    # apply different filters
    flt_image = convert_image_dtype(masked_batch, dtypes.float32)

    num_to_filter = int(batch_size*percent_filters)

    filter_inds = np.random.choice(np.arange(0,batch_size),num_to_filter,replace=False)
    filter_mask = np.zeros(batch_size)
    filter_mask[filter_inds] = 1
    filter_mask = filter_mask.astype('float32')
    inv_filter_mask = np.ones(batch_size)
    inv_filter_mask[filter_inds] = 0
    inv_filter_mask = inv_filter_mask.astype('float32')

    #
    hsv = gen_image_ops.rgb_to_hsv(flt_image)
    hue = array_ops.slice(hsv, [0, 0, 0, 0], [batch_size, -1, -1, 1])
    saturation = array_ops.slice(hsv, [0, 0, 0, 1], [batch_size, -1, -1, 1])
    value = array_ops.slice(hsv, [0, 0, 0, 2], [batch_size, -1, -1, 1])

    # hue
    delta_vals = random_ops.random_uniform([batch_size],-.15,.15)
    hue_deltas = tf.multiply(filter_mask,delta_vals)
    hue_deltas2 = tf.expand_dims(tf.transpose(tf.tile(tf.reshape(hue_deltas,[1,1,batch_size]),(crop_size[0],crop_size[1],1)),(2,0,1)),3)
    hue = math_ops.mod(hue + (hue_deltas2 + 1.), 1.)

    # saturation
    saturation_factor = random_ops.random_uniform([batch_size],.75,1.25)
    saturation_factor2 = tf.multiply(filter_mask,saturation_factor)
    saturation_factor3 = tf.add(inv_filter_mask,saturation_factor2)
    saturation_factor4 = tf.expand_dims(tf.transpose(tf.tile(tf.reshape(saturation_factor3,[1,1,batch_size]),(crop_size[0],crop_size[1],1)),(2,0,1)),3)

    saturation *= saturation_factor4
    saturation = clip_ops.clip_by_value(saturation, 0.0, 1.0)

    hsv_altered = array_ops.concat([hue, saturation, value], 3)
    rgb_altered = gen_image_ops.hsv_to_rgb(hsv_altered)

    # brightness
    brightness_factor = random_ops.random_uniform([batch_size],-.25,.25)
    brightness_factor2 = tf.multiply(filter_mask,brightness_factor)
    brightness_factor3 = tf.expand_dims(tf.transpose(tf.tile(tf.reshape(brightness_factor2,[1,1,batch_size]),(crop_size[0],crop_size[1],1)),(2,0,1)),3)
    adjusted = math_ops.add(rgb_altered,math_ops.cast(brightness_factor3,dtypes.float32))

    filtered_batch = clip_ops.clip_by_value(adjusted,0.0,1.0)

    # after we've doctored everything, we need to remember to subtract off the mean
    repMeanIm = np.tile(np.expand_dims(data.meanImage,0),[batch_size,1,1,1])
    final_batch = tf.subtract(filtered_batch,repMeanIm)

    print("Preparing network...")
    net = CaffeNetPlaces365({'data': final_batch})
    feat = net.layers[featLayer]

    expanded_a = tf.expand_dims(feat, 1)
    expanded_b = tf.expand_dims(feat, 0)
    D = tf.reduce_sum(tf.squared_difference(expanded_a, expanded_b), 2)

    D_mean, D_var = tf.nn.moments(D, axes=[0,1])
    bool_mask = tf.logical_and(D<=D_mean+tf.sqrt(D_var),D>=D_mean-tf.sqrt(D_var))

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

    loss = tf.multiply(mask,margin + posDistsRep - allDists)
    loss = tf.maximum(0., loss)
    loss = tf.reduce_mean(loss)

    # slightly counterintuitive to not define "init_op" first, but tf vars aren't known until added to graph
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    summary_op = tf.summary.merge_all()
    init_op = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver(max_to_keep=20)

    # Create data "batcher"
    data = CombinatorialTripletSet(filename, mean_file, img_size, crop_size, batch_size, num_pos_examples)

    # tf will consume any GPU it finds on the system. Following lines restrict it to "first" GPU
    c = tf.ConfigProto()
    c.gpu_options.visible_device_list="0,1"

    print("Starting session...")
    with tf.Session(config=c) as sess:
        sess.run(init_op)

        writer = tf.summary.FileWriter(log_dir, sess.graph)

        if pretrained_net is not None:
            net.load(pretrained_net, sess)

        print("Start training...")
        ctr  = 0
        for step in range(num_iters):
            start_time = time.time()
            batch, labels, ims = data.getBatch()
            people_masks = data.getPeopleMasks()
            _, loss_val = sess.run([train_op, loss], feed_dict={image_batch: batch, people_mask_batch: people_masks, label_batch: labels})
            dd = sess.run(D, feed_dict={image_batch: batch, people_mask_batch: people_masks, label_batch: labels})
            dd2 = sess.run(D2, feed_dict={image_batch: batch, people_mask_batch: people_masks, label_batch: labels})
            print len(np.where(dd==0)[0]),len(np.where(dd2==0)[0])

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
                checkpoint_file = os.path.join(ckpt_dir, 'checkpoint')
                saver.save(sess, checkpoint_file, global_step=step)

      #  coord.request_stop()
       # coord.join(threads)

if __name__ == "__main__":
    main()
