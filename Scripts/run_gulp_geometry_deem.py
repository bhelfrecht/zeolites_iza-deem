#!/usr/bin/env python

import os
import sys
import numpy as np
import subprocess

xyz_dir_base = '../Raw_Data/DEEM_330k/XYZ'
gulp_dir = '../Raw_Data/GULP/DEEM_330k/Geometry'
library_file = '../Raw_Data/GULP/catlow_mod.lib'
idxs_file = '../Processed_Data/Models/6.0/Linear_Models/'\
        'LSVC-LPCovR/4-Class/Power/OO+OSi+SiSi/GCH/rattled/vlist.idx'

n_iza = 225
idxs = np.loadtxt(idxs_file, dtype=int)
idxs = idxs[idxs >= n_iza] - n_iza
idxs_select_file = f'{gulp_dir}/optimize.idxs'
np.savetxt(idxs_select_file, idxs, fmt='%d')

p = subprocess.run([
    'python', 'gulp.py',
    f'{xyz_dir_base}/*/*.cif.xyz',
    gulp_dir,
    '-lf', library_file,
    '-kw', "'opti conp'",
    '-idxf', idxs_select_file
])

if not p:
    print('GULP calculations completed successfully!')
