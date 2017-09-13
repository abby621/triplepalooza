# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 11:49:13 2016

@author: souvenir
"""

import numpy as np
import cv2
import random
from doctor_ims import *
import glob
import socket
HOSTNAME = socket.gethostname()

# things we need to load for text insertion
if 'abby' in HOSTNAME.lower():
    fontDir = '/Users/abby/Documents/repos/fonts'
    peopleDir = '/Users/abby/Documents/datasets/people_crops'
else:
    fontDir = '/project/focus/datasets/fonts'
    peopleDir = '/project/focus/datasets/traffickcam/people_crops'

class CombinatorialTripletSet:
    def __init__(self, image_list, mean_file, image_size, crop_size, batch_size=100, num_pos=10, isTraining=True):
        self.image_size = image_size
        self.crop_size = crop_size

        self.meanFile = mean_file
        tmp = np.load(self.meanFile)/255.0
        meanIm = np.moveaxis(tmp, 0, -1)
        self.meanImage = cv2.resize(meanIm, (self.crop_size[0], self.crop_size[1]))

        #img = img - self.meanImage
        if len(self.meanImage.shape) < 3:
            self.meanImage = np.asarray(np.dstack((self.meanImage, self.meanImage, self.meanImage)))

        self.numPos = num_pos
        self.batchSize = batch_size

        self.files = []
        self.classes = []
        # Reads a .txt file containing image paths of image sets where each line contains
        # all images from the same set and the first image is the anchor
        f = open(image_list, 'r')
        ctr = 0
        for line in f:
            temp = line.strip('\n').split(' ')
            while len(temp) < self.numPos: # make sure we have at least 10 images available per class
                temp.append(random.choice(temp))
            self.files.append(temp)
            self.classes.append(ctr)
            ctr += 1

        self.isTraining = isTraining
        self.indexes = np.arange(0, len(self.files))

        self.people_crop_files = glob.glob(os.path.join(peopleDir,'*mask.png'))

    def getPeopleMasks(self):
        which_inds = random.sample(np.arange(0,len(self.people_crop_files)),self.batchSize)

        people_crops = np.zeros([self.batchSize,self.crop_size[0],self.crop_size[1]])
        for ix in range(0,self.batchSize):
            people_crops[ix,:,:] = self.getImageAsMask(self.people_crop_files[which_inds[ix]])

        people_crops = np.expand_dims(people_crops, axis=3)

        return people_crops

    def getBatch(self):
        numClasses = self.batchSize/self.numPos # need to handle the case where we need more classes than we have?
        classes = np.random.choice(self.classes,numClasses,replace=False)

        posClass = classes[0]
        random.shuffle(self.files[posClass])

        batch = np.zeros([self.batchSize, self.crop_size[0], self.crop_size[1], 3])

        labels = np.zeros([self.batchSize],dtype='int')
        ims = []

        for i in np.arange(self.numPos):
            if i < len(self.files[posClass]):
                img = self.getProcessedImage(self.files[posClass][i])
                if img is not None:
                    batch[i,:,:,:] = img
                labels[i] = posClass
                ims.append(self.files[posClass][i])

        ctr = self.numPos
        for negClass in classes[1:]:
            random.shuffle(self.files[negClass])
            for j in np.arange(self.numPos):
                if j < len(self.files[negClass]):
                    img = self.getProcessedImage(self.files[negClass][j])
                    if img is not None:
                        batch[ctr,:,:,:] = img
                    labels[ctr] = negClass
                    ims.append(self.files[negClass][j])
                ctr += 1

        return batch, labels, ims

    def getProcessedImage(self, image_file):
        img = cv2.imread(image_file)
        if img is None:
            return None

        img = img/255.0
        # if self.isTraining:
        #     img = doctor_im(img,ind)

        img = cv2.resize(img, (self.image_size[0], self.image_size[1]))

        if (self.isTraining):
            top = np.random.randint(self.image_size[0] - self.crop_size[0])
            left = np.random.randint(self.image_size[1] - self.crop_size[1])
            img = img[top:(top+self.crop_size[0]),left:(left+self.crop_size[1]),:]
        else:
            img = img[14:(self.crop_size[0] + 14), 14:(self.crop_size[1] + 14),:]

        return img

    def getImageAsMask(self, image_file):
        img = cv2.imread(image_file,cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        # how much of the image should the mask take up
        scale = np.random.randint(30,70)/float(100)
        resized_img = cv2.resize(img,(int(self.crop_size[0]*scale),int(self.crop_size[1]*scale)))

        # where should we put the mask?
        top = np.random.randint(0,self.crop_size[0]-resized_img.shape[0])
        left = np.random.randint(0,self.crop_size[1]-resized_img.shape[1])
        
        new_img = np.ones((self.crop_size[0],self.crop_size[1]))*255.0
        new_img[top:top+resized_img.shape[0],left:left+resized_img.shape[1]] = resized_img

        new_img[new_img<255] = 0
        new_img[new_img>1] = 1

        return new_img

class VanillaTripletSet:
    def __init__(self, image_list, image_size, crop_size, batch_size=100, num_pos=10, isTraining=True):
        self.meanFile = './models/mnist/mnist_mean.npy'
        tmp = np.load(self.meanFile)
        self.meanImage = np.moveaxis(tmp, 0, -1)
        if len(self.meanImage.shape) < 3:
            self.meanImage = np.asarray(np.dstack((self.meanImage, self.meanImage, self.meanImage)))

        self.numPos = num_pos
        self.batchSize = batch_size

        self.files = []
        self.classes = []
        # Reads a .txt file containing image paths of image sets where each line contains
        # all images from the same set and the first image is the anchor
        f = open(image_list, 'r')
        ctr = 0
        for line in f:
            temp = line[:-1].split(' ')
            while len(temp) < self.numPos: # make sure we have at least 10 images available per class
                temp.extend(random.choice(temp))
            self.files.append(temp)
            self.classes.append(ctr)
            ctr += 1

        self.image_size = image_size
        self.crop_size = crop_size
        self.isTraining = isTraining
        self.indexes = np.arange(0, len(self.files))

    def getBatch(self):
        numClasses = self.batchSize/3
        classes = np.random.choice(self.classes,numClasses)

        batch = np.zeros([self.batchSize, self.crop_size[0], self.crop_size[1], 3])
        labels = np.zeros([self.batchSize],dtype='int')
        dont_use_flag = np.zeros([self.batchSize],dtype='bool')

        ctr = 0
        for posClass in classes:
            random.shuffle(self.files[posClass])

            anchorIm = self.files[posClass][0]
            posIm = np.random.choice(self.files[posClass][1:])

            negClass = np.random.choice(self.classes)
            while negClass == posClass:
                negClass = np.random.choice(self.classes)
            random.shuffle(self.files[negClass])
            negIm = np.random.choice(self.files[negClass])

            anchorImg = self.getProcessedImage(anchorIm)
            posImg = self.getProcessedImage(posIm)
            negImg = self.getProcessedImage(negIm)

            batch[ctr,:,:,:] = anchorImg
            batch[ctr+1,:,:,:] = posImg
            batch[ctr+2,:,:,:] = negImg

            labels[ctr] = posClass
            labels[ctr+1] = posClass
            labels[ctr+2] = negClass
            ctr += 3

        return batch, labels

    def getProcessedImage(self, image_file):
        img = cv2.imread(image_file)
        img = cv2.resize(img, (self.image_size[0], self.image_size[1]))
        img = img - self.meanImage

        if (self.isTraining):
            top = np.random.randint(self.image_size[0] - self.crop_size[0])
            left = np.random.randint(self.image_size[1] - self.crop_size[1])
            img = img[top:(top+self.crop_size[0]),left:(left+self.crop_size[1]),:]
        else:
            img = img[14:(self.crop_size[0] + 14), 14:(self.crop_size[1] + 14),:]

        return img
