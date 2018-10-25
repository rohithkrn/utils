import pdb
import sys
import subprocess

file_path = "/pytorch/caffe2/core/hip/common_hip.h"

num_threads = [64, 128, 256, 32, 512]
num_blocks = [1024, 2048, 4096]

with open(file_path, 'r') as fin:
	src = fin.read()

prev_thread = 512
prev_block = 4096

for block in num_blocks:
	for threads in num_threads:

		src = src.replace("constexpr int CAFFE_HIP_NUM_THREADS = " + str(prev_thread),"constexpr int CAFFE_HIP_NUM_THREADS = " + str(threads))
		src = src.replace("constexpr int CAFFE_MAXIMUM_NUM_BLOCKS = " + str(prev_block), "constexpr int CAFFE_MAXIMUM_NUM_BLOCKS = " + str(block))

		prev_thread = threads
		prev_block = block

		with open(file_path, 'w') as fout:
			fout.write(src)

		build_log = "bl_" + str(block) + "_" + str(threads) + ".txt"
		with open(build_log, 'w') as out: 
			subprocess.call(["/bin/bash", "/pytorch/.jenkins/caffe2/amd/build_caffe2.sh"], stdout=out)

		bm_log = "bml_" + str(block) + "_" + str(threads) + ".txt"
		with open(bm_log, 'w') as out: 
			subprocess.call([sys.executable, 
				"/pytorch/build_caffe2/caffe2/python/convnet_benchmarks.py --batch_size 32 --model Resnet50"], 
				stdout=out)

