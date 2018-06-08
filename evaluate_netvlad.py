import tensorflow as tf
from classfile import NonTripletSet
import numpy as np
import netvlad_tf.net_from_mat as nfm
import netvlad_tf.nets as nets

test_file = '/project/focus/abby/triplepalooza/inputs/traffickcam/test.txt'
pretrained_net = '/project/focus/datasets/traffickcam/snapshots/lilou'
img_size = [256, 256]
crop_size = [227, 227]
mean_file = '/project/focus/abby/triplepalooza/models/traffickcam/tc_mean_im.npy'
batch_size = 120
test_data = NonTripletSet(test_file, mean_file, img_size, crop_size, batch_size, isTraining=False)

c = tf.ConfigProto()
c.gpu_options.visible_device_list="0"

tf.reset_default_graph()

image_batch = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
net_out = nets.vgg16NetvladPca(image_batch)
saver = tf.train.Saver()

sess = tf.Session(config=c)
saver.restore(sess, '/project/focus/abby/triplepalooza/models/netvlad/pretrained')

testingImsAndLabels = [(test_data.files[ix][iy],test_data.classes[ix]) for ix in range(len(test_data.files)) for iy in range(len(test_data.files[ix]))]
numTestingIms = batch_size*(len(testingImsAndLabels)/batch_size)
testingImsAndLabels = testingImsAndLabels[:numTestingIms]

testingFeats = np.empty((numTestingIms,net_out.shape[1]),dtype=np.float32)
testingIms = np.empty((numTestingIms),dtype=object)
testingLabels = np.empty((numTestingIms),dtype=np.int32)
for idx in range(0,numTestingIms,batch_size):
    print idx, '/', numTestingIms
    il = testingImsAndLabels[idx:idx+batch_size]
    ims = [i[0] for i in il]
    labels = [i[1] for i in il]
    batch = test_data.getBatchFromImageList(ims)
    batch = batch[...,::-1]
    testingIms[idx:idx+batch_size] = ims
    testingLabels[idx:idx+batch_size] = labels
    ff = sess.run(net_out, feed_dict={image_batch: batch})
    testingFeats[idx:idx+batch_size,:] = np.squeeze(ff)

def getDist(feat,otherFeats):
    dist = (otherFeats - feat)**2
    dist = np.sum(dist,axis=1)
    dist = np.sqrt(dist)
    return dist

def getDotDist(feat,otherFeats):
    dist = np.array([np.dot(feat,otherFeat) for otherFeat in otherFeats])
    return dist

print 'Computing testing set distances...'
queryImsAndLabels = [(testingIms[idx],testingLabels[idx],idx) for idx in range(numTestingIms) if 'resized_traffickcam' in testingIms[idx] or 'query' in testingIms[idx]]
testingAccuracy = np.zeros((len(queryImsAndLabels),100))
for idx in range(len(queryImsAndLabels)):
    thisIm = queryImsAndLabels[idx][0]
    thisLabel = queryImsAndLabels[idx][1]
    thisFeat = testingFeats[queryImsAndLabels[idx][2],:]
    dists = getDist(thisFeat,testingFeats)
    sortedInds = np.argsort(dists)[1:]
    sortedIms = testingIms[sortedInds]
    if 'query' in thisIm:
        bad = [aa for aa in range(len(sortedIms)) if 'query' in sortedIms[aa]]
        mask = np.ones(len(sortedInds), dtype=bool)
        mask[bad] = False
        sortedInds = sortedInds[mask,...]

    sortedIms = testingIms[sortedInds]
    if 'resized_traffickcam' in thisIm:
        thisTime = thisIm.split('/')[-1][:13]
        traffickCamInds = np.array([ix for ix in range(len(sortedIms)) if 'resized_traffickcam' in sortedIms[ix]])
        badInds = [ind for ind in traffickCamInds if sortedIms[ind].split('/')[-1][:13]==thisTime]
        mask = np.ones(sortedInds.size,dtype=bool)
        mask[badInds] = False
        sortedInds = sortedInds[mask]
        sortedIms = testingIms[sortedInds]

    sortedLabels = testingLabels[sortedInds]
    topHit = np.where(sortedLabels==thisLabel)[0][0]
    if topHit < 100:
        testingAccuracy[idx,topHit:] = 1

    if idx%10==0:
        print idx,': ',np.mean(testingAccuracy[:idx,:],axis=0)[0]
