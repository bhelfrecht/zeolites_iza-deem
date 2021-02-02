#!/usr/bin/env python

import os
import sys
import numpy as np
from gulp import gulp_summary

gulp_dir = '../Raw_Data/GULP/IZA_230'
gulp_glob = '*/*.out'
output = f'{gulp_dir}/optimization_summary.dat'

gulp_summary(output, gulp_dir, gulp_glob)
