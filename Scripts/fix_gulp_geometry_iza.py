#!/usr/bin/env python

import os
import sys
import subprocess
import glob
from gulp import fix_gulp

gulp_dir = '../Raw_Data/GULP/IZA_230'
summary_file = f'{gulp_dir}/optimization_summary.dat'
logfile = f'{gulp_dir}/fix_gulp.log'

fix_gulp(gulp_dir, summary_file, logfile)
