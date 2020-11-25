#!/usr/bin/env python

import os
import sys
import glob
import numpy as np
from tqdm import tqdm

ev_to_kJmol = 96.485307

cif_dir = '../Raw_Data/DEEM_330k/CIF'
gulp_dir = '../Raw_Data/GULP/DEEM_330k'
cif_files = sorted(glob.glob(f'{cif_dir}/*/*.cif'))
summary_file = f'{gulp_dir}/optimization_summary.dat'
summary_file_fix = f'{gulp_dir}/optimization_summary_fix.dat'

if os.path.exists(summary_file):
    g = open(summary_file_fix, 'w')
else:
    g = open(summary_file, 'w')

g.write('# ID | Database Energy | GULP Energy | '
        'GULP Gnorm | Minimum Failed | GULP Failed')
for cdx, cif_file in enumerate(tqdm(cif_files)):
    cif_energy = np.nan
    gulp_energy = np.nan
    gulp_gnorm = np.nan
    gulp_minimum_failed = 0 
    gulp_failed = 0

    basename = os.path.splitext(os.path.basename(cif_file))[0]
    structure_id = int(basename)

    gulp_file = f'{gulp_dir}/{basename}/{basename}.out'
    gulp_error_file = f'{gulp_dir}/{basename}/{basename}.log'

    if os.path.getsize(gulp_error_file) > 0:
        gulp_failed = 1 

    cif_target = 'GULP energy per Si atom'
    with open(cif_file, 'r') as cif:
        for line in cif:
            if cif_target in line:
                cif_energy = float(line.strip().split()[-2])
                break

    n = 0
    with open(gulp_file, 'r') as gulp:
        for line in gulp:
            if line.startswith('  Total number atoms/shells'):
                n = int(line.strip().split()[-1]) / 5
            elif 'Conditions for a minimum have not been satisfied' in line:
                gulp_minimum_failed = 1
            elif line.startswith('  Final energy') and n > 0:
                gulp_energy = ev_to_kJmol * float(line.strip().split()[-2]) / n
            elif line.startswith('  Final Gnorm'):
                gulp_gnorm = float(line.strip().split()[-1])
                break
    
    g.write(f'\n{structure_id:7d}  {cif_energy:15.8f}  {gulp_energy:15.8f}  '
            f'{gulp_gnorm:15.8f}  {gulp_minimum_failed:1d}  {gulp_failed:1d}')

g.close()
