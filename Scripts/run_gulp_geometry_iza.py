#!/usr/bin/env python

import os
import sys
import subprocess

xyz_dir = '../Raw_Data/IZA_230/XYZ'
gulp_dir = '../Raw_Data/GULP/IZA_230'
library_file = '../Raw_Data/GULP/catlow_mod.lib'

p = subprocess.run([
    'python', 'gulp.py',
    f'{xyz_dir}/*.xyz',
    gulp_dir,
    '-lf', library_file,
    '-kw', 'opti conp'
])

if not p:
    print('GULP calculations completed successfully!')
