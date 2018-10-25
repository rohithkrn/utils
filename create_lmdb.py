## create lmdb data-set

import numpy as np
from StringIO import StringIO
from caffe2.python import core, utils, workspace
from caffe2.proto import caffe2_pb2

import pandas as pd
import skimage.io
import skimage.transform
import lmdb
import pdb

data_root = '/data/test_models/distracted_driver_data/'

labels = pd.read_csv(data_root+'driver_imgs_list.csv').drop('subject', 1)[['img', 'classname']]
labels['img'] = labels.apply(lambda row: data_root+'imgs/train/'+row.classname+'/'+row.img, 1)
labels['classname'] = labels['classname'].map(lambda l: l[1])
labels = labels.reindex(np.random.permutation(labels.index))

labels.iloc[0:2000].to_csv(data_root+'valid.txt', sep=' ', header=False, index=False)
labels.iloc[2000:].to_csv(data_root+'train.txt', sep=' ', header=False, index=False)

train_input_file = data_root + 'train.txt'
train_db = data_root + 'data_train.minidb'
test_input_file = data_root + 'valid.txt'
test_db = data_root + 'data_test.minidb'

def write_db(db_type, db_name, input_file):
    db = core.C.create_db(db_type, db_name, core.C.Mode.write)
    transaction = db.new_transaction()
    with open(input_file,'r') as f:
    	data_paths = f.readlines()
    for j in range(len(data_paths)):
    	img_path = data_paths[j].split(' ')[0]
    	print(j,img_path)
        label = np.array(int(data_paths[j].split(' ')[1][0]))
        img = skimage.img_as_float(skimage.io.imread(img_path))
        img = skimage.transform.resize(img,(224,224))
        img = img[:,:,(2,1,0)]
        img_data = img.transpose(2,0,1)
        feature_and_label = caffe2_pb2.TensorProtos()
        feature_and_label.protos.extend([utils.NumpyArrayToCaffe2Tensor(img_data), utils.NumpyArrayToCaffe2Tensor(label)])
        transaction.put('train_%04d'.format(j),feature_and_label.SerializeToString())
    del transaction
    del db


if __name__ == "__main__":
    write_db("minidb",train_db,train_input_file)
    write_db("minidb",test_db,test_input_file)
    #create_db(train_lmdb,train_input_file)

