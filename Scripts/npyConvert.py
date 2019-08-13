#!/usr/bin/python

import os
import sys
import argparse
import numpy as np
import SOAPTools

parser = argparse.ArgumentParser()
parser.add_argument('-soap', type=str, 
                    help='File containing SOAP file filenames')
parser.add_argument('-convert', type=str, choices=['stdout', 'file'], default=None, 
                    help='Convert from .npy binary to ASCII')

args = parser.parse_args()

### CONVERT FROM .NPY TO .DAT ###
f = open(args.soap, 'r')
inputFiles = f.readlines()
inputFiles = [j.strip() for j in inputFiles]
f.close()
if args.convert == 'stdout':
    SOAPTools.npy_stdout(inputFiles)
else:
    SOAPTools.npy_convert(inputFiles)
