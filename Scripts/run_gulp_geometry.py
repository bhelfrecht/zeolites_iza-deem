#!/usr/bin/env python

import os
import sys
import numpy as np
import glob
from tqdm import tqdm
from gulp import cif2gulp, run_gulp

cif_dir = '../Raw_Data/DEEM_330k/CIF'
gulp_dir = '../Raw_Data/GULP/DEEM_330k/Geometry'
library_file = '../Raw_Data/GULP/catlow_mod.lib'
idxs_file = '../Processed_Data/Models/6.0/Linear_Models/'\
        'LSVC-LPCovR/4-Class/Power/OO+OSi+SiSi/GCH/rattled/vlist.idx'

n_iza = 225
idxs = np.loadtxt(idxs_file, dtype=int)
idxs = idxs[idxs >= n_iza] - n_iza
cif_files_glob = sorted(glob.glob(f'{cif_dir}/8*/*.cif'))
cif_files = [cif_files_glob[i] for i in idxs]

run_dir = os.getcwd()

if not os.path.exists(gulp_dir):
    os.makedirs(gulp_dir)

for cif_file in tqdm(cif_files):

    # Set up directories for GULP inputs and outputs
    basename = os.path.splitext(os.path.basename(cif_file))[0]
    gulp_run_dir = f'{gulp_dir}/{basename}'

    if not os.path.exists(gulp_run_dir):
        os.makedirs(gulp_run_dir)

    os.chdir(gulp_run_dir)
    gulp_input = f'{basename}.in'
    gulp_output = f'{basename}.out'
    gulp_log = f'{basename}.log'

    # Make GULP input files
    # IZA optimizations were done without symmetry, so we will do the same here
    cif2gulp(os.path.relpath(cif_file, start=gulp_run_dir), gulp_input,
            'opti conp nosymmetry', os.path.relpath(library_file, start=gulp_run_dir))
    
    # Run GULP
    run_gulp(gulp_input, gulp_output, gulp_log) 
    os.chdir(run_dir)
