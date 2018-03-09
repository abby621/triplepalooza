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
    def __init__(self, image_list, mean_file, image_size, crop_size, batchSize=100, num_pos=10, isTraining=True, isOverfitting=False):
        self.image_size = image_size
        self.crop_size = crop_size

        self.isTraining = isTraining
        self.isOverfitting = isOverfitting

        self.meanFile = mean_file
        meanIm = np.load(self.meanFile)

        if meanIm.shape[0] == 3:
            meanIm = np.moveaxis(meanIm, 0, -1)

        self.meanImage = cv2.resize(meanIm, (self.crop_size[0], self.crop_size[1]))

        #img = img - self.meanImage
        if len(self.meanImage.shape) < 3:
            self.meanImage = np.asarray(np.dstack((self.meanImage, self.meanImage, self.meanImage)))

        self.numPos = num_pos
        self.batchSize = batchSize

        self.files = []
        self.classes = []
        # Reads a .txt file containing image paths of image sets where each line contains
        # all images from the same set and the first image is the anchor
        f = open(image_list, 'r')
        ctr = 0
        for line in f:
            temp = line.strip('\n').split(' ')
            # if self.isTraining:
            #     while len(temp) < self.numPos: # make sure we have at least 10 images available per class
            #         temp.append(random.choice(temp))
            if len(temp) > self.numPos:
                self.files.append(temp)
                self.classes.append(ctr)
                ctr += 1

        # if we're overfitting, limit how much data we have per class
        if self.isOverfitting == True:
            self.classes = self.classes[:10]
            self.files = self.files[:10]
            for idx in range(len(self.files)):
                backupFiles = self.files[idx]
                self.files[idx] = backupFiles[:10]

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

        batch = np.zeros([self.batchSize, self.crop_size[0], self.crop_size[1], 3])

        labels = np.zeros([self.batchSize],dtype='int')
        ims = []

        ctr = 0
        for cls in classes:
            random.shuffle(self.files[cls])
            for j in np.arange(self.numPos):
                if j < len(self.files[cls]):
                    img = self.getProcessedImage(self.files[cls][j])
                    if img is not None:
                        batch[ctr,:,:,:] = img
                    labels[ctr] = cls
                    ims.append(self.files[cls][j])
                ctr += 1

        return batch, labels, ims

    def getBatchFromImageList(self,image_list):
        batch = np.zeros([len(image_list), self.crop_size[0], self.crop_size[1], 3])
        for ix in range(0,len(image_list)):
            img = self.getProcessedImage(image_list[ix])
            batch[ix,:,:,:] = img
        return batch

    def getProcessedImage(self, image_file):
        img = cv2.imread(image_file)
        if img is None:
            return None

        if self.isTraining and not self.isOverfitting and random.random() > 0.5:
            img = cv2.flip(img,1)

        # if self.isTraining:
        #     img = doctor_im(img)

        img = cv2.resize(img, (self.image_size[0], self.image_size[1]))

        if self.isTraining and not self.isOverfitting:
            top = np.random.randint(self.image_size[0] - self.crop_size[0])
            left = np.random.randint(self.image_size[1] - self.crop_size[1])
        else:
            top = int(round((self.image_size[0] - self.crop_size[0])/2))
            left = int(round((self.image_size[1] - self.crop_size[1])/2))

        img = img[top:(top+self.crop_size[0]),left:(left+self.crop_size[1]),:]

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

class MarsCombinatorialTripletSet(CombinatorialTripletSet):
    def getBatch(self):
        numClasses = self.batchSize/self.numPos # need to handle the case where we need more classes than we have?
        classes = np.random.choice(self.classes,numClasses,replace=False)

        batch = np.zeros([self.batchSize, self.crop_size[0], self.crop_size[1], 3])

        labels = np.zeros([self.batchSize],dtype='int')
        cameras = np.empty([self.batchSize],dtype='int')
        ims = []

        ctr = 0
        for cls in classes:
            possibleCams = np.unique([self.files[cls][ix].split('/')[-1].split('.')[0][4:6] for ix in range(len(self.files[cls]))])
            camsSeen = []
            j = 0
            while j < self.numPos:
                # if we have to double back, we're going to have to allow duplicate entries in the batch
                if len(camsSeen) == len(possibleCams):
                    camsSeen = []

                random.shuffle(self.files[cls])
                im = self.files[cls][0]

                person_info = im.split('/')[-1].split('.')[0]
                person_id = person_info[:4]
                camera = person_info[5:6]
                tracklet = person_info[6:11]
                frame = person_info[11:]
                if camera not in camsSeen:
                    camsSeen.append(camera)
                    img = self.getProcessedImage(im)
                    if img is not None:
                        batch[ctr,:,:,:] = img
                    labels[ctr] = cls
                    cameras[ctr] = int(camera)
                    ims.append(im)
                    j += 1
                    ctr += 1

        return batch, labels, cameras, ims

class VanillaTripletSet:
    def __init__(self, image_list, mean_file, image_size, crop_size, batchSize=100, isTraining=True, isOverfitting=False, isMixed=False):
        self.image_size = image_size
        self.crop_size = crop_size
        self.isOverfitting = isOverfitting
        self.isMixed = isMixed
        self.meanFile = mean_file
        meanIm = np.load(self.meanFile)

        if meanIm.shape[0] == 3:
            meanIm = np.moveaxis(meanIm, 0, -1)

        self.meanImage = cv2.resize(meanIm, (self.crop_size[0], self.crop_size[1]))

        #img = img - self.meanImage
        if len(self.meanImage.shape) < 3:
            self.meanImage = np.asarray(np.dstack((self.meanImage, self.meanImage, self.meanImage)))

        self.batchSize = batchSize

        self.files = []
        self.classes = []
        # Reads a .txt file containing image paths of image sets where each line contains
        # all images from the same set and the first image is the anchor
        f = open(image_list, 'r')
        ctr = 0
        for line in f:
            temp = line[:-1].split(' ')
            self.files.append(temp)
            self.classes.append(ctr)
            ctr += 1

        if self.isOverfitting == True:
            self.classes = self.classes[:10]
            self.files = self.files[:10]
            for idx in range(len(self.files)):
                backupFiles = self.files[idx]
                self.files[idx] = backupFiles[:10]

        self.image_size = image_size
        self.crop_size = crop_size
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

    def getBatch(self):
        numClasses = self.batchSize/3
        classes = np.zeros(numClasses,dtype=np.int)
        badClasses = []
        selectedClasses = 0
        while selectedClasses < numClasses:
            cls = np.random.choice(self.classes)
            while cls in classes or cls in badClasses:
                cls = np.random.choice(self.classes)

            files = self.files[cls]
            num_traffickcam = np.sum([1 for fl in files if 'resized_traffickcam' in fl])
            num_expedia = np.sum([1 for fl in files if 'resized_expedia' in fl])

            if num_traffickcam > 3 and num_expedia > 3:
                classes[selectedClasses] = cls
            else:
                badClasses.append(cls)

            selectedClasses += 1

        batch = np.zeros([self.batchSize, self.crop_size[0], self.crop_size[1], 3])
        labels = np.zeros([self.batchSize],dtype='int')
        ims = []
        dont_use_flag = np.zeros([self.batchSize],dtype='bool')

        ctr = 0
        for posClass in classes:
            random.shuffle(self.files[posClass])
            anchorIm = self.files[posClass][0]
            anchorImg = self.getProcessedImage(anchorIm)
            while anchorImg is None:
                random.shuffle(self.files[posClass])
                anchorIm = self.files[posClass][0]
                anchorImg = self.getProcessedImage(anchorIm)

            posIm = np.random.choice(self.files[posClass][1:])
            if self.isMixed:
                while ('expedia' in anchorIm and 'expedia' in posIm) or ('expedia' not in anchorIm and 'expedia' not in posIm):
                    posIm = np.random.choice(self.files[posClass][1:])

            posImg = self.getProcessedImage(posIm)
            while posImg is None:
                posIm = np.random.choice(self.files[posClass][1:])
                while posIm == anchorIm or ('expedia' in anchorIm and 'expedia' in posIm) or ('expedia' not in anchorIm and 'expedia' not in posIm):
                    posIm = np.random.choice(self.files[posClass][1:])
                posImg = self.getProcessedImage(posIm)

            negClass = np.random.choice(self.classes)
            while negClass == posClass:
                negClass = np.random.choice(self.classes)

            random.shuffle(self.files[negClass])
            negIm = np.random.choice(self.files[negClass])
            negImg = self.getProcessedImage(negIm)
            while negImg is None:
                negIm = np.random.choice(self.files[negClass])
                negImg = self.getProcessedImage(negIm)

            batch[ctr,:,:,:] = anchorImg
            batch[ctr+1,:,:,:] = posImg
            batch[ctr+2,:,:,:] = negImg

            labels[ctr] = posClass
            labels[ctr+1] = posClass
            labels[ctr+2] = negClass

            ims.append(anchorIm)
            ims.append(posIm)
            ims.append(negIm)

            ctr += 3

        return batch, labels, ims

    def getBatchFromImageList(self,image_list):
        batch = np.zeros([len(image_list), self.crop_size[0], self.crop_size[1], 3])
        for ix in range(0,len(image_list)):
            img = self.getProcessedImage(image_list[ix])
            batch[ix,:,:,:] = img

        return batch

    def getProcessedImage(self, image_file):
        img = cv2.imread(image_file)
        if img is None:
            return img

        if self.isTraining and random.random() > 0.5 and not self.isOverfitting:
            img = cv2.flip(img,1)

        img = cv2.resize(img, (self.image_size[0], self.image_size[1]))

        if self.isTraining and not self.isOverfitting:
            top = np.random.randint(self.image_size[0] - self.crop_size[0])
            left = np.random.randint(self.image_size[1] - self.crop_size[1])
        else:
            top = int(round((self.image_size[0] - self.crop_size[0])/2))
            left = int(round((self.image_size[1] - self.crop_size[1])/2))

        img = img[top:(top+self.crop_size[0]),left:(left+self.crop_size[1]),:]
        return img

class MixedSetTripletSet(VanillaTripletSet):
    def getBatch(self):
        numClasses = self.batchSize/9
        classes = np.zeros(numClasses,dtype=np.int)
        badClasses = []
        selectedClasses = 0
        while selectedClasses < numClasses:
            cls = np.random.choice(self.classes)
            while cls in classes or cls in badClasses:
                cls = np.random.choice(self.classes)

            files = self.files[cls]
            num_traffickcam = np.sum([1 for fl in files if 'resized_traffickcam' in fl])
            num_expedia = np.sum([1 for fl in files if 'resized_expedia' in fl])

            if num_traffickcam > 5 and num_expedia > 5:
                classes[selectedClasses] = cls
            else:
                badClasses.append(cls)

            selectedClasses += 1

        np.random.shuffle(classes)

        batch = np.zeros([self.batchSize, self.crop_size[0], self.crop_size[1], 3])
        labels = np.zeros([self.batchSize],dtype='int')
        ims = []
        dont_use_flag = np.zeros([self.batchSize],dtype='bool')

        ctr = 0
        for posClass in classes:
            random.shuffle(self.files[posClass])
            anchorIm = self.files[posClass][0]
            anchorImg = self.getProcessedImage(anchorIm)
            while anchorImg is None:
                random.shuffle(self.files[posClass])
                anchorIm = self.files[posClass][0]
                anchorImg = self.getProcessedImage(anchorIm)

            batch[ctr,:,:,:] = anchorImg
            labels[ctr] = posClass
            ims.append(anchorIm)
            ctr += 1

            for pos_idx in range(4):
                posIm = np.random.choice(self.files[posClass][1:])
                if self.isMixed:
                    while ('expedia' in anchorIm and 'expedia' in posIm) or ('expedia' not in anchorIm and 'expedia' not in posIm):
                        posIm = np.random.choice(self.files[posClass][1:])

                posImg = self.getProcessedImage(posIm)
                while posImg is None:
                    posIm = np.random.choice(self.files[posClass][1:])
                    while posIm == anchorIm or ('expedia' in anchorIm and 'expedia' in posIm) or ('expedia' not in anchorIm and 'expedia' not in posIm):
                        posIm = np.random.choice(self.files[posClass][1:])
                    posImg = self.getProcessedImage(posIm)

                batch[ctr,:,:,:] = posImg
                labels[ctr] = posClass
                ims.append(posIm)
                ctr += 1

            for neg_idx in range(4):
                negClass = np.random.choice(self.classes)
                while negClass == posClass:
                    negClass = np.random.choice(self.classes)

                random.shuffle(self.files[negClass])
                negIm = np.random.choice(self.files[negClass])
                negImg = self.getProcessedImage(negIm)
                while negImg is None:
                    negIm = np.random.choice(self.files[negClass])
                    negImg = self.getProcessedImage(negIm)

                batch[ctr,:,:,:] = negImg
                labels[ctr] = negClass
                ims.append(negIm)
                ctr += 1

        return batch, labels, ims

class NonTripletSet:
    def __init__(self, image_list, mean_file, image_size, crop_size, batchSize=100, isTraining=True):
        self.image_size = image_size
        self.crop_size = crop_size

        self.meanFile = mean_file
        meanIm = np.load(self.meanFile)

        if meanIm.shape[0] == 3:
            meanIm = np.moveaxis(meanIm, 0, -1)

        self.meanImage = cv2.resize(meanIm, (self.crop_size[0], self.crop_size[1]))

        #img = img - self.meanImage
        if len(self.meanImage.shape) < 3:
            self.meanImage = np.asarray(np.dstack((self.meanImage, self.meanImage, self.meanImage)))

        self.batchSize = batchSize

        self.files = []
        self.classes = []
        # Reads a .txt file containing image paths of image sets where each line contains
        # all images from the same set and the first image is the anchor
        f = open(image_list, 'r')
        ctr = 0
        for line in f:
            temp = line[:-1].split(' ')
            self.files.append(temp)
            self.classes.append(ctr)
            ctr += 1

        self.image_size = image_size
        self.crop_size = crop_size
        self.isTraining = isTraining
        self.indexes = np.arange(0, len(self.files))

    def getBatch(self):
        batch = np.zeros([self.batchSize, self.crop_size[0], self.crop_size[1], 3])
        labels = np.zeros([self.batchSize],dtype='int')
        ims = np.zeros([self.batchSize],dtype=object)

        for ix in range(0,self.batchSize):
            randClass = random.choice(self.classes)
            randIm = random.choice(self.files[randClass])
            randImg = self.getProcessedImage(randIm)
            while randImg is None:
                randClass = np.random.choice(self.classes)
                randIm = random.choice(self.files[randClass])
                randImg = self.getProcessedImage(randIm)

            batch[ix,:,:,:] = randImg
            labels[ix] = randClass
            ims[ix] = randIm

        return batch, labels, ims

    def getBatchFromImageList(self,image_list):
        batch = np.zeros([len(image_list), self.crop_size[0], self.crop_size[1], 3])
        for ix in range(0,len(image_list)):
            img = self.getProcessedImage(image_list[ix])
            batch[ix,:,:,:] = img

        return batch

    def getProcessedImage(self, image_file):
        img = cv2.imread(image_file)
        if img is None:
            return img

        if self.isTraining and random.random() > 0.5:
            img = cv2.flip(img,1)

        img = cv2.resize(img, (self.image_size[0], self.image_size[1]))

        if (self.isTraining):
            top = np.random.randint(self.image_size[0] - self.crop_size[0])
            left = np.random.randint(self.image_size[1] - self.crop_size[1])
        else:
            top = int(round((self.image_size[0] - self.crop_size[0])/2))
            left = int(round((self.image_size[1] - self.crop_size[1])/2))

        img = img[top:(top+self.crop_size[0]),left:(left+self.crop_size[1]),:]
        return img

class MarsNonTripletSet(NonTripletSet):
    def getBatch(self):
        batch = np.zeros([self.batchSize, self.crop_size[0], self.crop_size[1], 3])
        labels = np.zeros([self.batchSize],dtype='int')
        cams = np.zeros([self.batchSize],dtype='int')
        ims = np.zeros([self.batchSize],dtype=object)

        for ix in range(0,self.batchSize):
            randClass = random.choice(self.classes)
            randIm = random.choice(self.files[randClass])

            person_info = randIm.split('/')[-1].split('.')[0]
            person_id = person_info[:4]
            camera = person_info[5:6]
            tracklet = person_info[6:11]
            frame = person_info[11:]

            cams[ix] = camera

            randImg = self.getProcessedImage(randIm)
            while randImg is None:
                randClass = np.random.choice(self.classes)
                randIm = random.choice(self.files[randClass])
                randImg = self.getProcessedImage(randIm)

            batch[ix,:,:,:] = randImg
            labels[ix] = randClass
            ims[ix] = randIm

        return batch, labels, cams, ims

    def getBatchFromImageList(self,image_list):
        batch = np.zeros([len(image_list), self.crop_size[0], self.crop_size[1], 3])
        cams = np.zeros([len(image_list)],dtype='int')
        for ix in range(0,len(image_list)):
            person_info = image_list[ix].split('/')[-1].split('.')[0]
            person_id = person_info[:4]
            camera = person_info[5:6]
            tracklet = person_info[6:11]
            frame = person_info[11:]

            img = self.getProcessedImage(image_list[ix])
            batch[ix,:,:,:] = img
            cams[ix] = camera

        return batch, cams
