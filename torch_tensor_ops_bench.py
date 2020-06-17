import torch
import torch.nn as nn

import time
import argparse

dtype_map = {"fp32" : torch.float32, 
			 "fp16" : torch.float16,
			  "bf16" : torch.bfloat16}

binary_ops = ['add', 'mul', 'div', 'sub', 'eq']
unary_ops = ['exp', 'relu', 'tanh', 'sqrt']
reduction_ops = ['sum', 'prod', 'norm', 'max', 'mean', 'std', 'var', 'argmax', 'argmin']
all_ops = binary_ops + unary_ops + reduction_ops

def time_wrap(use_gpu):
    if use_gpu:
        torch.cuda.synchronize()
    return time.time()

def benchmark(args):
	device = torch.device(args.device)
	dtype = dtype_map[args.dtype]
	input_dim = [int(dim) for dim in args.input_dim.split("-")]
	ops = all_ops if args.run_all else [args.op]
	input1 = torch.randn(input_dim, device=device, dtype=dtype)
	input2 = torch.randn(input_dim, device=device, dtype=dtype)
	for op_str in ops:
		input1 = torch.randn(input_dim, device=device, dtype=dtype)
		input2 = torch.randn(input_dim, device=device, dtype=dtype)
		op_args = []
		if op_str in binary_ops or args.op_type == 'binary':
			op_args.append(input2)
		if ((op_str in binary_ops) or (op_str in unary_ops)) and op_str[-1] != '_':
			op_str += '_'  #inplace
		op = getattr(input1, op_str)
		for _ in range(args.num_warmup_iters):
			op(*op_args)

		with torch.autograd.profiler.profile(args.enable_profiling, args.use_gpu) as prof:
			start_time = time_wrap(args.use_gpu)
			for _ in range(args.num_iters):
				op(*op_args)
			end_time = time_wrap(args.use_gpu)
			time_per_iter = 1000.0*(end_time - start_time)/args.num_iters
			print("time per iter for {} : {:.2f} ms/it".format(op_str, time_per_iter))

		if args.enable_profiling:
			if args.use_gpu:
				print(prof.key_averages().table(sort_by="cuda_time_total"))
			else:
				print(prof.key_averages().table(sort_by="cpu_time_total"))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--device", default='cuda', required=False, type=str)
	parser.add_argument("--dtype", default='fp32', required=False, type=str, choices=['fp32', 'fp16', 'bf16'])
	parser.add_argument("--input-dim", default="64-1024-1024", type=str, required=False)
	parser.add_argument("--op", default='add', required=False, type=str)
	parser.add_argument("--op-type", default=None, required=False, type=str)
	parser.add_argument("--run-all", default=False, action="store_true")
	parser.add_argument("--num-iters", default=20, type=int, required=False)
	parser.add_argument("--num-warmup-iters", default=5, type=int, required=False)
	parser.add_argument("--enable-profiling", action="store_true", default=False)

	args = parser.parse_args()
	args.use_gpu = True if 'cuda' in args.device else False
	benchmark(args)