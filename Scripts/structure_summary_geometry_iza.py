#!/usr/bin/env python

import os
import sys
import glob
from ase.io import read, write
from gulp import structure_summary
from project_utils import removeprefix, removesuffix, get_basename

ref_dir = '../Raw_Data/IZA_230/XYZ'
ref_ext = 'xyz'
gulp_dir = '../Raw_Data/GULP/IZA_230'
gulp_glob = '*/*_opt.cif'
geometry_file = f'{gulp_dir}/geometry_errors.dat'

structure_summary(geometry_file, gulp_dir, gulp_glob, ref_dir, ref_ext)

# Concatenate the optimized geometries
cif_files = sorted(glob.glob(f'{gulp_dir}/{gulp_glob}'))
frames = []
for cif_file in cif_files:
    iza_code = removesuffix(get_basename(cif_file), '_opt')
    frame = read(cif_file, format='cif')
    frame.info['code'] = iza_code
    frames.append(frame)

write(f'{gulp_dir}/IZA_230.xyz', frames, format='extxyz')
