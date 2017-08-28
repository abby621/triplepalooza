import glob
import os

classes = ['0','1','2','3','4','5','6','7','8','9']

dataset = 'mnist'
test_or_train = 'train'
datadir = os.path.join('/Users/abby/Documents/datasets/',dataset,test_or_train)

output_folder = os.path.join('/Users/abby/Documents/repos/tf_tons_of_triplets/inputs/',dataset)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

output_file_path = os.path.join(output_folder,test_or_train+'.txt')
if os.path.exists(output_file_path):
    os.remove(output_file_path)

ims = {}
with open(output_file_path,'a') as output_file:
    for cls in classes:
        ims[cls] = glob.glob(os.path.join(datadir,cls,'*.jpg'))
        im_str = ' '.join(ims[cls])
        output_file.write(im_str+'\n')
