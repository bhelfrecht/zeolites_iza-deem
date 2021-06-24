#!/usr/bin/env python

import os
import sys
import subprocess
from gulp import gulp_summary, fix_gulp

xyz_dir = '../Raw_Data/Alpha_Quartz'
gulp_dir = '../Raw_Data/GULP/Alpha_Quartz'
library_file = '../Raw_Data/GULP/catlow_mod.lib'
summary_file = f'{gulp_dir}/optimization_summary.dat'
logfile = f'{gulp_dir}/fix_gulp.log'

# Initial constant pressure GULP calculation
p = subprocess.run([
    'python', 'gulp.py',
    f'{xyz_dir}/alpha_quartz.xyz',
    gulp_dir,
    '-lf', library_file,
    '-kw', 'opti conp'
])

# Extract energy and check convergence
gulp_summary(summary_file, gulp_dir, 'alpha_quartz/alpha_quartz.out')

# If not converged, run constant volume GULP calculation
fix_gulp(gulp_dir, summary_file, logfile)

# Extract energy (will duplicate the summary file
# if the constant pressure calculation is converged)
gulp_summary(summary_file, gulp_dir, 'alpha_quartz/alpha_quartz.out')
