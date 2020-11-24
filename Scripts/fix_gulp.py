#!/usr/bin/env python

import os
import sys
import subprocess
import glob
from tqdm import tqdm
from gulp import run_gulp

# TODO: make log file for fixing and write debug info
gulp_dir = '../Raw_Data/GULP/DEEM_330k'
summary_file = f'{gulp_dir}/optimization_summary.dat'
run_dir = os.getcwd()

def backup_run(files, backup_dir):
    os.mkdir(backup_dir)
    for f in files:
        os.rename(f, f'{backup_dir}/{f}')

def replace_input_line(old_file, new_file, old_line, new_line):
    g = open(new_file, 'w')
    with open(old_file, 'r') as f:
        for line in f:
            if old_line in line:
                g.write(line.replace(old_line, new_line))
            else:
                g.write(line)
    g.close()

with open(summary_file, 'r') as f:
    for line in tqdm(f):

        if line.startswith('#'):
            continue

        line_data = line.strip().split()
        structure_id = line_data[0]
        failed_minimum = int(line_data[-2])
        failed = int(line_data[-1])

        if failed or failed_minimum:
            os.chdir(f'{gulp_dir}/{structure_id}')

            failed_dir = 'SYM'
            failed_files = glob.glob('{structure_id}.*')
            backup_run(failed_files, failed_dir)

            new_gulp_input = f'{structure_id}.in'
            new_gulp_output = f'{structure_id}.out'
            new_gulp_log = f'{structure_id}.log'
            replace_input_line(f'{failed_dir}/{structure_id}.in', new_gulp_input, 
                    'opti conv shell', 'opti conv shell nosymmetry')
            run_gulp(new_gulp_input, new_gulp_output, new_gulp_log)

            os.chdir(run_dir)
            break
