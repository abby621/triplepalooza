# -*- coding: utf-8 -*-
"""
# python tc_setloss.py margin batch_size output_size learning_rate whichGPU is_finetuning l1_weight bn_decay
# chop off last layer: python tc_setloss.py .3 126 128 .0001 '3' True 0.05 0.9
# don't chop off last layer: python tc_setloss.py .3 126 128 .0001 '3' False 0.05 0.9
"""

import tensorflow as tf
from classfile import MixedSetTripletSet
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
from nets import resnet_v2
import socket
import signal
import time
import sys

def main(margin,batch_size,output_size,learning_rate,whichGPU,is_finetuning,l1_weight,bn_decay):
    def handler(signum, frame):
        print 'Saving checkpoint before closing'
        pretrained_net = os.path.join(ckpt_dir, 'checkpoint-'+param_str)
        saver.save(sess, pretrained_net, global_step=step)
        print 'Checkpoint-',pretrained_net+'-'+str(step), ' saved!'
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)

    ckpt_dir = './output/traffickcam/ckpts/finetuning'
    log_dir = './output/traffickcam/logs'
    train_filename = './inputs/traffickcam/train.txt'
    mean_file = './models/traffickcam/tc_mean_im.npy'
    pretrained_net = '/project/focus/abby/triplepalooza/models/ilsvrc-2012/resnet_v2_50.ckpt'
    # pretrained_net = './output/traffickcam/ckpts/finetuning/checkpoint-2018_03_08_1004_lr1e-06_outputSz128_margin0pt4_l1wgt0pt05_bndecay0pt9-20880'
    img_size = [256, 256]
    crop_size = [224, 224]
    num_iters = 200000
    summary_iters = 10
    save_iters = 5000
    featLayer = 'resnet_v2_50/logits'

    is_training = True

    margin = float(margin)
    batch_size = int(batch_size)
    output_size = int(output_size)
    learning_rate = float(learning_rate)
    l1_weight = float(l1_weight)
    batch_norm_decay = float(bn_decay)

    if batch_size%9 != 0:
        print 'Batch size must be divisible by 9 (1 anchor, 4 +, 4 -)!'
        sys.exit(0)

    # Create data "batcher"
    train_data = MixedSetTripletSet(train_filename, mean_file, img_size, crop_size, batch_size, isTraining=is_training, isMixed=True)
    numClasses = len(train_data.files)
    numIms = np.sum([len(train_data.files[idx]) for idx in range(0,numClasses)])
    datestr = datetime.now().strftime("%Y_%m_%d_%H%M")
    param_str = datestr+'_lr'+str(learning_rate).replace('.','pt')+'_outputSz'+str(output_size)+'_margin'+str(margin).replace('.','pt')+'_l1wgt'+str(l1_weight).replace('.','pt')+'_bndecay'+str(batch_norm_decay).replace('.','pt')
    logfile_path = os.path.join(log_dir,param_str+'_train.txt')
    train_log_file = open(logfile_path,'a')
    print '------------'
    print ''
    print 'Going to train with the following parameters:'
    print '# Classes: ',numClasses
    train_log_file.write('# Classes: '+str(numClasses)+'\n')
    print '# Ims: ',numIms
    train_log_file.write('# Ims: '+str(numIms)+'\n')
    print 'Margin: ',margin
    train_log_file.write('Margin: '+str(margin)+'\n')
    print 'Output size: ', output_size
    train_log_file.write('Output size: '+str(output_size)+'\n')
    print 'Learning rate: ',learning_rate
    train_log_file.write('Learning rate: '+str(learning_rate)+'\n')
    print 'Logging to: ',logfile_path
    train_log_file.write('Param_str: '+param_str+'\n')
    train_log_file.write('----------------\n')
    print ''
    print '------------'

    # Queuing op loads data into input tensor
    image_batch = tf.placeholder(tf.float32, shape=[batch_size, crop_size[0], crop_size[0], 3])
    people_mask_batch = tf.placeholder(tf.float32, shape=[batch_size, crop_size[0], crop_size[0], 1])
    label_batch = tf.placeholder(tf.int32, shape=(batch_size))

    # doctor image params
    percent_crop = .5
    percent_people = .5
    percent_rotate = .2
    percent_filters = .4
    percent_text = .1

    # # richard's argument: since the data is randomly loaded, we don't need to change the indices that we perform operations on every time; i am on board with this, but had already implemented the random crops, so will leave that for now
    # # apply random rotations
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
    # shuffled_inds = np.arange(0,batch_size,dtype='int32')
    # np.random.shuffle(shuffled_inds)
    crop_inds = tf.slice(shuffled_inds,[0],[num_to_crop])
    uncropped_inds = tf.slice(shuffled_inds,[num_to_crop],[num_to_not_crop])

    # crop_ratio = float(3)/float(5)
    # crop_yx = tf.random_uniform([num_to_crop,2], 0,1-crop_ratio, dtype=tf.float32, seed=0)
    # crop_sz = tf.add(crop_yx,np.tile([crop_ratio,crop_ratio],[num_to_crop, 1]))
    # crop_boxes = tf.concat([crop_yx,crop_sz],axis=1)

    # randomly select a crop between 3/5 of the image and the entire image
    crop_ratio = tf.random_uniform([num_to_crop,1], float(3)/float(5), 1, dtype=tf.float32, seed=0)
    # randomly select a starting location between 0 and the max valid x position
    crop_yx = tf.random_uniform([1,2],0.,1.-crop_ratio, dtype=tf.float32,seed=0)
    crop_sz = tf.add(crop_yx,tf.concat([crop_ratio,crop_ratio],axis=1))
    crop_boxes = tf.concat([crop_yx,crop_sz],axis=1)

    uncropped_boxes = np.tile([0,0,1,1],[num_to_not_crop,1])

    all_inds = tf.concat([crop_inds,uncropped_inds],axis=0)
    all_boxes = tf.concat([crop_boxes,uncropped_boxes],axis=0)

    sorted_inds = tf.nn.top_k(-shuffled_inds,sorted=True,k=batch_size).indices
    cropped_batch = tf.gather(tf.image.crop_and_resize(rotated_batch,all_boxes,all_inds,crop_size),sorted_inds)

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
    # hue = math_ops.mod(hue + (hue_deltas2 + 1.), 1.)
    hue_mod = tf.add(hue,hue_deltas2)
    hue = clip_ops.clip_by_value(hue_mod,0.0,1.0)

    # saturation
    saturation_factor = random_ops.random_uniform([batch_size],-.05,.05)
    saturation_factor2 = tf.multiply(filter_mask,saturation_factor)
    saturation_factor3 = tf.expand_dims(tf.transpose(tf.tile(tf.reshape(saturation_factor2,[1,1,batch_size]),(crop_size[0],crop_size[1],1)),(2,0,1)),3)
    saturation_mod = tf.add(saturation,saturation_factor3)
    saturation = clip_ops.clip_by_value(saturation_mod, 0.0, 1.0)

    hsv_altered = array_ops.concat([hue, saturation, value], 3)
    rgb_altered = gen_image_ops.hsv_to_rgb(hsv_altered)

    # brightness
    brightness_factor = random_ops.random_uniform([batch_size],-.25,.25)
    brightness_factor2 = tf.multiply(filter_mask,brightness_factor)
    brightness_factor3 = tf.expand_dims(tf.transpose(tf.tile(tf.reshape(brightness_factor2,[1,1,batch_size]),(crop_size[0],crop_size[1],1)),(2,0,1)),3)
    adjusted = math_ops.add(rgb_altered,math_ops.cast(brightness_factor3,dtypes.float32))

    filtered_batch = clip_ops.clip_by_value(adjusted,0.0,255.0)

    # after we've doctored everything, we need to remember to subtract off the mean
    repMeanIm = np.tile(np.expand_dims(train_data.meanImage,0),[batch_size,1,1,1])
    noise = tf.random_normal(shape=[batch_size, crop_size[0], crop_size[0], 1], mean=0.0, stddev=0.0025, dtype=tf.float32)
    final_batch = tf.add(tf.subtract(filtered_batch,repMeanIm),noise)

    # after we've doctored everything, we need to remember to subtract off the mean
    # repMeanIm = np.tile(np.expand_dims(train_data.meanImage,0),[batch_size,1,1,1])
    # if train_data.isOverfitting:
    #     final_batch = tf.subtract(masked_batch,repMeanIm)
    # else:
    #     noise = tf.random_normal(shape=[batch_size, crop_size[0], crop_size[0], 1], mean=0.0, stddev=0.0025, dtype=tf.float32)
    #     final_batch = tf.add(tf.subtract(masked_batch,repMeanIm),noise)

    print("Preparing network...")
    with slim.arg_scope(resnet_v2.resnet_arg_scope(updates_collections=None, batch_norm_decay=batch_norm_decay)):
        _, layers = resnet_v2.resnet_v2_50(final_batch, num_classes=output_size, is_training=True)

    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        if is_finetuning.lower() == 'true' and var.op.name.startswith('resnet_v2_50/logits') or 'momentum' in var.op.name.lower():
            excluded = True
        if not excluded:
            variables_to_restore.append(var)

    feat = tf.squeeze(tf.nn.l2_normalize(layers[featLayer],3))
    idx = tf.range(0, batch_size, 9)
    posIdx = tf.reshape(tf.tile(tf.expand_dims(idx,1),[1,4]),[4*batch_size/9]) + tf.tile(tf.range(1,5),[batch_size/9])
    negIdx = tf.reshape(tf.tile(tf.expand_dims(idx,1),[1,4]),[4*batch_size/9]) + tf.tile(tf.range(5,9),[batch_size/9])

    ancFeat = tf.expand_dims(tf.gather(feat, idx),1)
    ancFeats = tf.reshape(tf.tile(ancFeat,(1,4,1)),[batch_size/9*4,output_size])
    posFeats = tf.gather(feat, posIdx)
    negFeats = tf.gather(feat, negIdx)

    dPos = tf.reshape(tf.reduce_sum(tf.square(posFeats - ancFeats), 1),(batch_size/9,4))
    dNeg = tf.reshape(tf.reduce_sum(tf.square(negFeats - ancFeats), 1),(batch_size/9,4))

    loss = tf.maximum(0., margin + tf.reduce_min(dPos,axis=0) - tf.reduce_min(dNeg,axis=0))
    loss = tf.reduce_mean(loss)

    # slightly counterintuitive to not define "init_op" first, but tf vars aren't known until added to graph
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = slim.learning.create_train_op(loss, optimizer)

    summary_op = tf.summary.merge_all()
    init_op = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver(max_to_keep=20)

    # tf will consume any GPU it finds on the system. Following lines restrict it to specific gpus
    c = tf.ConfigProto()
    if not 'abby' in socket.gethostname().lower():
        c.gpu_options.visible_device_list=whichGPU

    print("Starting session...")
    sess = tf.Session(config=c)
    sess.run(init_op)

    writer = tf.summary.FileWriter(log_dir, sess.graph)

    restore_fn = slim.assign_from_checkpoint_fn(pretrained_net,variables_to_restore)
    restore_fn(sess)

    print("Start training...")
    ctr  = 0
    for step in range(num_iters):
        start_time = time.time()
        batch, labels, ims = train_data.getBatch()
        people_masks = train_data.getPeopleMasks()
        _, loss_val = sess.run([train_op, loss], feed_dict={image_batch: batch, people_mask_batch: people_masks,label_batch: labels})
        end_time = time.time()
        duration = end_time-start_time
        out_str = 'Step %d: loss = %.6f -- (%.3f sec)' % (step, loss_val, duration)
        print(out_str)
        if step % summary_iters == 0:
            # print(out_str)
            train_log_file.write(out_str+'\n')
        # Update the events file.
        # summary_str = sess.run(summary_op)
        # writer.add_summary(summary_str, step)
        # writer.flush()
        #
        # Save a checkpoint
        if (step + 1) % save_iters == 0:
            print('Saving checkpoint at iteration: %d' % (step))
            pretrained_net = os.path.join(ckpt_dir, 'checkpoint-'+param_str)
            saver.save(sess, pretrained_net, global_step=step)
            print 'checkpoint-',pretrained_net+'-'+str(step), ' saved!'
        if (step + 1) == num_iters:
            print('Saving final')
            pretrained_net = os.path.join(ckpt_dir, 'final-'+param_str)
            saver.save(sess, pretrained_net, global_step=step)
            print 'final-',pretrained_net+'-'+str(step), ' saved!'

    sess.close()
    train_log_file.close()

if __name__ == "__main__":
    args = sys.argv
    if len(args) < 9:
        print 'Expected input parameters: margin,batch_size,output_size,learning_rate,whichGPU,is_finetuning,l1_weight,bn_decay'
    margin = args[1]
    batch_size = args[2]
    output_size = args[3]
    learning_rate = args[4]
    whichGPU = args[5]
    is_finetuning = args[6]
    l1_weight = args[7]
    bn_decay = args[8]
    main(margin,batch_size,output_size,learning_rate,whichGPU,is_finetuning,l1_weight,bn_decay)
