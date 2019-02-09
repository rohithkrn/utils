## script to run benchmarks

import os
import subprocess
import re
import pandas
import argparse

from collections import OrderedDict

## Dictionary which maps model and corresponding 'batch_size's 
## to run benchmarks. Comment the models you do not want to run. 
model_bs_map = OrderedDict([
			 ('AlexNet', [1, 32, 64, 256]),
			 ('VGGA', [1, 32, 64]),
			 ('Inception', [1, 32, 64, 128]),
			 ('Resnet50', [1, 32, 64]),
			 ('Resnet101', [1, 32]),
			 ('Resnext101', [1, 32]),
			 ])

## set this path to convnet_benchmarks_dpm.py
script_path = "convnet_benchmarks_dpm.py"


def get_benchmark_cmd(args):

	benchmark_cmd = "python " + script_path + " --model " + args.model + \
						" --num_gpus " + str(args.num_gpus) + \
						" --batch_size " + str(args.batch_size)
	log_file = args.model + "_" + str(args.batch_size/args.num_gpus) + \
					"_" + str(args.num_gpus) + 'g'

	if args.dtype == "float16":
		benchmark_cmd += " --dtype float16"
		log_file += "_fp16"

	if args.forward_only:
		benchmark_cmd += " --forward_only"
		log_file += "_forward"

	if args.layer_wise_benchmark:
		benchmark_cmd += " --net_type simple --layer_wise_benchmark"

	return benchmark_cmd, log_file


def run_benchmark_cmd(benchmark_cmd, log_file, batch_size, layer_wise):

	with open(log_file, 'w+') as out:
		if layer_wise:
			ret_code = subprocess.call(benchmark_cmd.split(' '), stdout=out)
		else:
			ret_code = subprocess.call(benchmark_cmd.split(' '), stderr=out)

	if ret_code == 0:
		with open(log_file, 'r') as src:
			in_src = src.read()
		
		match=re.search("Milliseconds per iter:\s+(\d+\.\d+)", in_src)
		ms_per_iter = float(match.group(1))
		images_per_sec = batch_size*1000/ms_per_iter
		return images_per_sec, ret_code
	else:
		print("\n >>>>>> The following benchmark FAILED to run:\n" + benchmark_cmd)
		return None, ret_code

def run_sgpu_benchmarks(args):

	results_dict = {}
	failing_configs = []

	for model in model_bs_map.keys():

		single_model_results = {}
		args.model = model

		for batch_size in model_bs_map[model]:

			args.batch_size = batch_size
			benchmark_cmd, log_file = get_benchmark_cmd(args)
			log_file += ".log"
			log_file_path = os.path.join(args.logs_dir, "logs")
			
			if not os.path.exists(log_file_path):
				os.makedirs(log_file_path)

			log_file = os.path.join(log_file_path, log_file)

			print("\n>>>>>>>>>> Running config >>>>>>>>")
			print(benchmark_cmd + "\n")

			images_per_sec, ret_code = run_benchmark_cmd(benchmark_cmd,
														 log_file,
														 args.batch_size,
														 args.layer_wise_benchmark)
			
			if ret_code == 0:
				single_model_results[batch_size] = images_per_sec
			else:
				failing_configs.append(benchmark_cmd)

		results_dict[model] = single_model_results

	# print failing configs:
	if len(failing_configs) > 0:
		with open(os.path.join(args.logs_dir, "failed_configs.log"), 'w+') as fc:
			print("######## FAILING CONFIGS ##########")
			for config in failing_configs:
				print(config)
				fc.write(config+'\n')
		
	print("##### Results - Batch Size vs Imgs/sec #####")	
	df = pandas.DataFrame(results_dict)
	print(df)

	csv_file_path =  os.path.join(args.logs_dir, 'csv')

	if not os.path.exists(csv_file_path):
		os.makedirs(csv_file_path)

	csv_file = os.path.join(csv_file_path, "results_sgpu.csv")	
	df.to_csv(csv_file)


def run_mgpu_benchmarks(args):

	failing_configs = []

	for model in model_bs_map.keys():

		results_dict = {}
		args.model = model

		for num_gpus in range(1,5): # 1, 2, 3, 4 gpus

			args.num_gpus = num_gpus
			results_with_bs = {}

			for batch_size in model_bs_map[model]:

				args.batch_size = num_gpus*batch_size
				benchmark_cmd, log_file = get_benchmark_cmd(args)
				log_file += ".log"
				log_file_path = os.path.join(args.logs_dir, "logs")
			
				if not os.path.exists(log_file_path):
					os.makedirs(log_file_path)

				log_file = os.path.join(log_file_path, log_file)

				print("\n>>>>>>>>>> Running config >>>>>>>>>>>")
				print(benchmark_cmd + "\n")

				images_per_sec, ret_code = run_benchmark_cmd(benchmark_cmd,
														 	 log_file,
														 	 args.batch_size,
														 	 args.layer_wise_benchmark)

				if ret_code == 0:
					results_with_bs[batch_size] = images_per_sec
				else:
					failing_configs.append(benchmark_cmd)

			results_dict[num_gpus] = results_with_bs

		print("##### Results for {} #####".format(model))
		df = pandas.DataFrame(results_dict)
		print(df)

		csv_file_path =  os.path.join(args.logs_dir, 'csv')

		if not os.path.exists(csv_file_path):
			os.makedirs(csv_file_path)

		csv_file = os.path.join(csv_file_path, model + "_mgpu.csv")	
		df.to_csv(csv_file)

	# print failing configs:
	if len(failing_configs) > 0:
		with open(os.path.join(args.logs_dir, "failed_configs.log"), 'w+') as fc:
			print("######## FAILING CONFIGS ##########")
			for config in failing_configs:
				print(config)
				fc.write(config+'\n')	

if __name__ == "__main__":

	parser = argparse.ArgumentParser(
        description="Runs Caffe2 convnet benchmarks.")
	parser.add_argument("--logs_dir", type=str, default=None, required=True,
                        help="Directory path to store logs/results.")
	parser.add_argument("--dtype", type=str, default='float', 
    					choices=["float", "float16"])
	parser.add_argument("--num_gpus", type=int, default=1,
    					help="Number of gpus to run benchmarks.")
	parser.add_argument("--forward_only", action='store_true')
	parser.add_argument("--layer_wise_benchmark", action='store_true')
	parser.add_argument("--mgpu", action='store_true',
    					help="Set this to run on 1,2,3,4 gpus to analyze scaling.")

	args = parser.parse_args()

	if os.path.exists(args.logs_dir):
		print("logs_dir already exists. Provide an alternate directory")
		exit(1)
	else:
		os.makedirs(args.logs_dir)
		
	if args.mgpu:
		run_mgpu_benchmarks(args)
	else:					
   		run_sgpu_benchmarks(args)
