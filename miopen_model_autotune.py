import sys
import os

from collections import OrderedDict

inception_path = '/data/models/inception_v2'

model_bs_map = OrderedDict([
	('Resnet50',(32, 64)),
	('Resnext101', (32, 64)),
	('AlexNet', (256, 512, 1024)),
	('VGGA', (64, 128)),
	('Inception_v2', (32, 64))
])


for model, bs_list in model_bs_map.items():
	if model is not "Inception_v2":
		for bs in bs_list:
			print model, bs
			os.system("MIOPEN_FIND_ENFORCE=3 pytorch/build_caffe2/caffe2/python/convnet_benchmarks.py --batch_size " \
					+ str(bs) + "--model " + model)
	else:
		for bs in bs_list:
			print model, bs
			os.system("MIOPEN_FIND_ENFORCE=3 /pytorch/build_caffe2/caffe2/python/convnet_benchmarks.py --batch_size " \
					+ str(bs) + " --model " + model + " --model_path " + inception_path)


