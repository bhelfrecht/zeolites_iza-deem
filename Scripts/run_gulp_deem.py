#!/usr/bin/env python

import os
import sys
import subprocess

xyz_dir_base = '../Raw_Data/DEEM_330k/XYZ'
xyz_dirs = [
   '800-804', 
   '805-809', 
   '810-814', 
   '815-819', 
   '820-824', 
   '825-829', 
   '830-833'
]
gulp_dir = '../Raw_Data/GULP/DEEM_330k'

# Library file location 
library_file = '../Raw_Data/GULP/catlow_mod.lib'

processes = []
for xyz_dir in xyz_dirs:
    p = subprocess.Popen([
        'python', 'gulp.py',
        f'{xyz_dir_base}/{xyz_dir}/*.cif.xyz',
        gulp_dir,
        '-lf', library_file,
        '-kw', 'opti conv shell'
    ])
    processes.append(p)

exit_codes = [p.wait() for p in processes]
if not all(exit_codes):
    print('GULP calculations completed successfully!')
