#!/usr/bin/env python

import os
import sys
import subprocess

cif_dir_base = '../Raw_Data/DEEM_330k/CIF'
cif_dirs = [
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
for cif_dir in cif_dirs:
    p = subprocess.Popen([
        'python', 'gulp.py',
        f'{cif_dir_base}/{cif_dir}',
        f'{gulp_dir}',
        '-l', f'{library_file}'
        ])
    processes.append(p)

exit_codes = [p.wait() for p in processes]
if not all(exit_codes):
    print('GULP calculations completed successfully!')
