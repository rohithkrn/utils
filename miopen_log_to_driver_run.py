# script to parse MIOPEN_ENABLE_LOGGING_CMD=1 log
# and run the driver configs to measure performance
# Run the script from miopen build directory, i.e 
# it exepcts ./bin/MIOpenDriver to be there in build 
# directory

import argparse
import os
import subprocess

import pandas as pd
from collections import OrderedDict

failing_configs = []

conv_driver_dict_list = []
conv_total_forward_time = 0.0
conv_total_backward_time = 0.0
conv_log_map = {'MIOpenDriver': 'driver_command',
				'MIOpen Forward Conv. Algorithm': 'forward_algo',
				'GPU Kernel Time Forward Conv. Elapsed': 'forward_time',
				'MIOpen Backward Data Conv. Algorithm': 'backward_data_algo',
				'GPU Kernel Time Backward Data Conv. Elapsed': 'backward_weight_time',
				'MIOpen Backward Weights Conv. Algorithm': 'backward_weight_algo',
				'GPU Kernel Time Backward Weights Conv. Elapsed': 'backward_data_time',
                # add bias mapping here
			   }

bnorm_driver_dict_list = []
bnorm_total_forward_time = 0.0
bnorm_total_backward_time = 0.0
bnorm_log_map = {'MIOpenDriver': 'driver_command',
                 'GPU Kernel Min Time Forward Batch Normalization Elapsed': 'forward_time',
                 'GPU Kernel Min Time Backwards Batch Normalization Elapsed': 'backward_time',
                }

def run_driver(driver_cmd):
    log_file = 'tmp.txt'
    with open(log_file, 'w+') as out:
        subprocess.call(driver_cmd.split(" "), stdout=out)

    with open(log_file, 'r') as dri_out:
        driver_output = dri_out.readlines()

    os.remove(log_file)
    return driver_output

def check_failure(driver_cmd, driver_output):
    global failing_configs
    driver_output_join = " ".join(driver_output)
    if "Fail" in driver_output_join or "dumped" in driver_output_join:
        failing_configs.append(driver_cmd)
        print "the above command failed"
        return False
    else:
        return True

def process_conv(driver_cmd):
    global conv_driver_dict_list
    global conv_total_forward_time
    global conv_total_backward_time

    print driver_cmd

    driver_output = run_driver(driver_cmd)

    if not check_failure(driver_cmd, driver_output):
        return

    output_dict = OrderedDict()
    for line in driver_output:  
        line_split = line.split(":")
        if len(line_split) == 2:
            output_dict[conv_log_map[line_split[0]]] = line_split[1].strip()
    forward_time = float(output_dict['forward_time'].strip().split(" ")[0])
    backward_weight_time = float(output_dict['backward_weight_time'].strip().split(" ")[0])
    backward_data_time = float(output_dict['backward_data_time'].strip().split(" ")[0])
    backward_time = backward_data_time + backward_weight_time
    print "(forward_time, backward_time):", (round(forward_time, 5), round(backward_time, 5))
    conv_total_forward_time += forward_time
    conv_total_backward_time += backward_time

    ## TODO: if bias present case

    output_dict['forward_time'] = forward_time
    output_dict['backward_weight_time'] = backward_weight_time
    output_dict['backward_data_time'] = backward_data_time
    conv_driver_dict_list.append(output_dict)

def process_bnorm(driver_cmd):
    global bnorm_driver_dict_list
    global bnorm_total_forward_time
    global bnorm_total_backward_time

    driver_cmd += " -t 1"
    print driver_cmd
    driver_output = run_driver(driver_cmd)

    if not check_failure(driver_cmd, driver_output):
        return

    output_dict = OrderedDict()
    for line in driver_output:  
        line_split = line.split(":")
        if len(line_split) == 2:
            output_dict[bnorm_log_map[line_split[0]]] = line_split[1].strip()

    forward_time = float(output_dict['forward_time'].strip().split(" ")[0])
    output_dict['forward_time'] = forward_time
    bnorm_total_forward_time += forward_time
    print "forward_time:", (round(forward_time, 5))

    driver_cmd += " -b 1"
    print driver_cmd

    driver_output = run_driver(driver_cmd)

    if not check_failure(driver_cmd, driver_output):
        return

    for line in driver_output:  
        line_split = line.split(":")
        if len(line_split) == 2:
            output_dict[bnorm_log_map[line_split[0]]] = line_split[1].strip()

    backward_time = float(output_dict['backward_time'].strip().split(" ")[0])
    output_dict['backward_time'] = backward_time
    print "backward_time:", (round(forward_time, 5))
    bnorm_total_backward_time += backward_time

    bnorm_driver_dict_list.append(output_dict)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Runs miopen driver from MIOPEN_ENABLE_LOGGING_CMD=1"
    )
    
    parser.add_argument("--log_file", type=str, default=None, required=True,
                        help="Path to miopen log file")
    
    args = parser.parse_args()
    
    with open(args.log_file,'r') as infile:
        lines = infile.readlines()

    miopen_lines = []
    for line in lines:
        if line[:6] == "miopen":
            miopen_lines.append(line)

    supported_ops = {'conv': process_conv, 'bnorm': process_bnorm}
    driver_cmd_set = set()

    for line in miopen_lines:
        line_split = line.split(':')
        miopen_call = line_split[0]
        driver_cmd = line_split[1].strip()
        miopen_op = driver_cmd.split(" ")[1]

        if miopen_op not in supported_ops:
            print("op-type {} not supported".format(miopen_op))

        if driver_cmd in driver_cmd_set:
            continue
        else:
            driver_cmd_set.add(driver_cmd)
        if "Forward" in miopen_call:
            supported_ops[miopen_op](driver_cmd)


    conv_out_df = pd.DataFrame(conv_driver_dict_list)
    conv_out_df.to_csv("conv_test.csv", index=False)

    bnorm_out_df = pd.DataFrame(bnorm_driver_dict_list)
    bnorm_out_df.to_csv("bnorm_test.csv", index=False)

    if len(failing_configs) > 0:
        print "###### Failing configs #######"
        for cfg in failing_configs:
            print cfg

    print "============ Total Times ================"
    print "conv total_forward_time:", conv_total_forward_time
    print "conv total_backward_time:", conv_total_backward_time

    print "bnorm total_forward_time:", bnorm_total_forward_time
    print "bnorm total_backward_time:", bnorm_total_backward_time

