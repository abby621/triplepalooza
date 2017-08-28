import glob
import os, csv

machine = 'focus'

if machine == 'local':
    expedia_path = '/Users/abby/Documents/datasets/resized_expedia/'
    traffickcam_path = '/Users/abby/Documents/datasets/resized_traffickcam/'
    expedia_im_file = '/Users/abby/Documents/datasets/resized_expedia/current_expedia_ims.txt'
    traffickcam_im_file = '/Users/abby/Documents/datasets/resized_traffickcam/current_traffickcam_ims.txt'
    output_folder = '/Users/abby/Documents/datasets//triplepalooza/'
else:
    expedia_path = '/project/focus/datasets/traffickcam/resized_expedia/'
    traffickcam_path = '/project/focus/datasets/traffickcam/resized_traffickcam/'
    expedia_im_file = '/project/focus/datasets/traffickcam/current_expedia_ims.txt'
    traffickcam_im_file = '/project/focus/datasets/traffickcam/current_traffickcam_ims.txt'
    output_folder = '/project/focus/datasets/traffickcam/triplepalooza/'

with open(traffickcam_im_file,'rU') as f:
    rd = csv.reader(f,delimiter='\t')
    traffickcam_ims = list(rd)

traffickcam_ims.pop(0)

im_list = []
flipped_im_list = []
for im_id,hotel_id,im in traffickcam_ims:
    new_path = im.replace('/mnt/EI_Code/ei_code/django_ei/submissions/',traffickcam_path)
    if os.path.exists(new_path):
        im_list.append((new_path,hotel_id))
    flipped_path = new_path.replace('.jpg','_flipped.jpg')
    if os.path.exists(flipped_path):
        flipped_im_list.append((flipped_path,hotel_id))

with open(expedia_im_file,'rU') as f:
    rd = csv.reader(f,delimiter='\t')
    expedia_ims = list(rd)

expedia_ims.pop(0)

for im_id,hotel_id,im in expedia_ims:
    new_path = os.path.join(expedia_path,str(hotel_id),str(im_id)+'.jpg')
    if os.path.exists(new_path):
        im_list.append((new_path,hotel_id))
    flipped_path = new_path.replace('.jpg','_flipped.jpg')
    if os.path.exists(flipped_path):
        flipped_im_list.append((flipped_path,hotel_id))

ims_by_class = {}
for i in im_list:
    if not i[1] in ims_by_class:
        ims_by_class[i[1]] = {}
        ims_by_class[i[1]]['traffickcam'] = []
        ims_by_class[i[1]]['expedia'] = []
    if 'expedia' in i[0]: # expedia
        if not i[0] in ims_by_class[i[1]]['expedia']:
            ims_by_class[i[1]]['expedia'].append(i[0])
    else: # traffickcam
        if not i[0] in ims_by_class[i[1]]['traffickcam']:
            ims_by_class[i[1]]['traffickcam'].append(i[0])

classes = ims_by_class.keys()
numClassesStart = len(classes)
numIms = 0
for cls in classes:
    if len(ims_by_class[cls]['traffickcam']) == 0 or len(ims_by_class[cls]['expedia']) == 0:
        ims_by_class.pop(cls, None)
    else:
        numIms += len(ims_by_class[cls]['traffickcam'])
        numIms += len(ims_by_class[cls]['expedia'])

classes = ims_by_class.keys()
numClasses = len(classes)

allClasses = np.zeros((numIms),dtype='int')
allIms = []
tcOrExpedia = np.zeros((numIms),dtype='str')
startInd = 0
for cls in classes:
    for captureType in ['traffickcam','expedia']:
        allClasses[startInd:startInd+len(ims_by_class[cls][captureType])] = int(cls)
        tcOrExpedia[startInd:startInd+len(ims_by_class[cls][captureType])] = captureType[0]
        allIms.extend(ims_by_class[cls][captureType])
        startInd += len(ims_by_class[cls][captureType])

classes_0_ind = {}
for ix in range(0,len(classes)):
    classes_0_ind[classes[ix]] = ix

setAsideClasses = []
setAsideClasses_0ind = []
setAsideQueries = []
setAsideDb = []
while len(setAsideClasses) < 500:
    cls = random.choice(classes)
    posInds = np.where(allClasses==int(cls))[0]
    if cls not in setAsideClasses and 't' in tcOrExpedia[posInds] and 'e' in tcOrExpedia[posInds]:
        class_0_ind = classes_0_ind[cls]
        setAsideClasses.append(cls)
        tcIms = posInds[np.where(tcOrExpedia[posInds]=='t')[0]]
        queryImInd = random.choice(tcIms)
        setAsideQueries.append((allIms[queryImInd],class_0_ind))
        exIms = posInds[np.where(tcOrExpedia[posInds]=='e')[0]]
        for imInd in random.sample(exIms,min(10,len(exIms))):
            setAsideDb.append((allIms[imInd],class_0_ind))

while len(setAsideClasses) < 1000:
    cls = random.choice(classes)
    if cls not in setAsideClasses and 'e' in tcOrExpedia[posInds]:
        class_0_ind = classes_0_ind[cls]
        setAsideClasses.append(cls)
        exIms = posInds[np.where(tcOrExpedia[posInds]=='e')[0]]
        for imInd in random.sample(exIms,min(10,len(exIms))):
            setAsideDb.append((allIms[imInd],class_0_ind))

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

test_path = os.path.join(output_folder,'test.txt')
if os.path.exists(test_path):
    os.remove(test_path)

with open(test_path,'a') as test_file:
    for cls in setAsideClasses:
        these_ims = ims_by_class[cls]['traffickcam'] + ims_by_class[cls]['expedia']
        im_str = ' '.join(these_ims)
        test_file.write(im_str+'\n')

train_path = os.path.join(output_folder,'train.txt')
if os.path.exists(train_path):
    os.remove(train_path)

with open(train_path,'a') as train_file:
    for cls in classes:
        if cls not in setAsideClasses:
            these_ims = ims_by_class[cls]['traffickcam'] + ims_by_class[cls]['expedia']
            im_str = ' '.join(these_ims)
            train_file.write(im_str+'\n')
