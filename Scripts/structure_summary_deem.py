#!/usr/bin/env python

import os
import sys
from gulp import structure_summary

ref_dir = '../Raw_Data/DEEM_330k/XYZ'
ref_ext = 'cif.xyz'
gulp_dir = '../Raw_Data/GULP/DEEM_330k'
gulp_glob = '*/*_opt.cif'
geometry_file = f'{gulp_dir}/geometry_errors.dat'

structure_summary(geometry_file, gulp_dir, gulp_glob, ref_dir, ref_ext)
