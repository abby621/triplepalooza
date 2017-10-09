import glob, os

train_ims = glob.glob('/project/focus/hong/Embeding/MARS/tra/*/*.jpg')
test_ims = glob.glob('/project/focus/hong/Embeding/MARS/val/*/*.jpg')

trainImsByClass = {}
for im in train_ims:
    cls = im.split('/')[-2]
    if not cls in trainImsByClass:
        trainImsByClass[cls] = []
    trainImsByClass[cls].append(im)

testImsByClass = {}
for im in test_ims:
    cls = im.split('/')[-2]
    if not cls in testImsByClass:
        testImsByClass[cls] = []
    testImsByClass[cls].append(im)

output_folder = '/project/focus/abby/triplepalooza/inputs/mars/'

train_path = os.path.join(output_folder,'train.txt')
if os.path.exists(train_path):
    os.remove(train_path)

with open(train_path,'a') as train_file:
    for cls in trainImsByClass.keys():
        these_ims = trainImsByClass[cls]
        im_str = ' '.join(these_ims)
        train_file.write(im_str+'\n')

test_path = os.path.join(output_folder,'test.txt')
if os.path.exists(test_path):
    os.remove(test_path)

with open(test_path,'a') as test_file:
    for cls in testImsByClass.keys():
        these_ims = testImsByClass[cls]
        im_str = ' '.join(these_ims)
        test_file.write(im_str+'\n')
