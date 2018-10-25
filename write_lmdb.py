import numpy as np
import os
import lmdb
from imageio import imread
from caffe2.proto import caffe2_pb2

training_labels_path = "/home/messi/data/train_label_tmp.txt"

def write_lmdb(labels_file_path, lmdb_path):
    labels_handler = open(labels_file_path, "r")
    # Write to lmdb
    print(">>> Write database...")
    LMDB_MAP_SIZE = 1 << 40
    print("LMDB_MAP_SIZE", LMDB_MAP_SIZE)
    env = lmdb.open(lmdb_path, map_size=LMDB_MAP_SIZE)

    with env.begin(write=True) as txn:
        count = 0
        for line in labels_handler.readlines():
            line = line.rstrip()
            im_path = line.split()[0]
            im_label = int(line.split()[1])
            
            # read in image (as RGB)
            img_data = imread(im_path).astype(np.float32)
            
            # convert to BGR
            img_data = img_data[:, :, (2, 1, 0)]
            
            # HWC -> CHW (N gets added in AddInput function)
            img_data = np.transpose(img_data, (2,0,1))
            
            # Create TensorProtos
            tensor_protos = caffe2_pb2.TensorProtos()
            img_tensor = tensor_protos.protos.add()
            img_tensor.dims.extend(img_data.shape)
            img_tensor.data_type = 1
            flatten_img = img_data.reshape(np.prod(img_data.shape))
            img_tensor.float_data.extend(flatten_img)
            label_tensor = tensor_protos.protos.add()
            label_tensor.data_type = 2
            label_tensor.int32_data.append(im_label)
            txn.put(
                '{}'.format(count).encode('ascii'),
                tensor_protos.SerializeToString()
            )
            if ((count % 2 == 0)):
                print("Inserted {} rows".format(count))
            count = count + 1

    print("Inserted {} rows".format(count))
    print("\nLMDB saved at " + lmdb_path + "\n\n")
    labels_handler.close()

training_lmdb_path = "/home/messi/data/lmdb/in_train.lmdb"
# Call function to write our LMDBs
if not os.path.exists(training_lmdb_path):
    print("Writing training LMDB")
    write_lmdb(training_labels_path, training_lmdb_path)
else:
    print(training_lmdb_path, "already exists!")
