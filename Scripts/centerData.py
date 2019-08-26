#!/usr/bin/python

import os
import sys
import argparse
import numpy as np
import SOAPTools

parser = argparse.ArgumentParser()
parser.add_argument('-soap', type=str, 
                    help='File containing SOAP file filenames')

args = parser.parse_args()

# Center the data
SOAPTools.center_data(args.soap)

