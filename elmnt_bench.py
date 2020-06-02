import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dtype', type=str, required=False, default='fp32')
parser.add_argument('--size', type=int, required=False, default=10000)
parser.add_argument('--iters', type = int, required=False, default=100)
args = parser.parse_args()

dtype_dict = {'fp32' : torch.float32,
              'fp16' : torch.float16,
              'bf16' : torch.bfloat16}

dtype = dtype_dict[args.dtype]
iters = args.iters

device = torch.device("cuda:0")

x = torch.rand(args.size, device=device, dtype=dtype)
y = torch.rand(args.size, device=device, dtype=dtype)

# warmup 
for _ in range(5):
   z = x + y

with torch.autograd.profiler.profile(True, True) as prof:
    for _ in range(iters):
        z = x + y

print(prof.key_averages().table(sort_by="cuda_time_total"))

