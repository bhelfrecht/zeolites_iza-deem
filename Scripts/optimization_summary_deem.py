#!/usr/bin/env python

import os
import sys
import numpy as np
from gulp import gulp_summary

ref_dir = '../Raw_Data/DEEM_330k/CIF'
ref_ext = 'cif'
gulp_dir = '../Raw_Data/GULP/DEEM_330k'
gulp_glob = '*/*.out'
output = f'{gulp_dir}/optimization_summary.dat'

gulp_summary(output, gulp_dir, gulp_glob, ref_ext=ref_ext, ref_dir=ref_dir)
