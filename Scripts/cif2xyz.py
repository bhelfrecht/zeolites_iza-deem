#!/usr/bin/env python

import os
import sys
from ase.io import read, write
from ase.spacegroup.spacegroup import SpacegroupNotFoundError
from tqdm import tqdm
import glob
from tempfile import TemporaryFile
import shlex
import argparse
from project_utils import removeprefix, removesuffix, get_basename

# From ase.io.cif:
# some spacegroups will not be recognized
# with their 'older' names,
# so they must be converted
spacegroup_name_change = {
    'A b m 2': 'A e m 2', 
    'A b a 2': 'A e a 2', 
    'C m c a': 'C m c e', 
    'C m m a': 'C m m e', 

    # Conversions from GULP-compatible spacegroups
    'P m 3': 'P m -3',
    'P n 3': 'P n -3',
    'F m 3': 'F m -3',
    'F d 3': 'F d -3',
    'I m 3': 'I m -3',
    'P a 3': 'P a -3',
    'I a 3': 'I a -3',
    'P m 3 m': 'P m -3 m',
    'P n 3 n': 'P n -3 n',
    'P m 3 n': 'P m -3 n',
    'P n 3 m': 'P n -3 m',
    'F m 3 m': 'F m -3 m',
    'F m 3 c': 'F m -3 c',
    'F d 3 m': 'F d -3 m',
    'F d 3 c': 'F d -3 c',
    'I m 3 m': 'I m -3 m',
    'I a 3 d': 'I a -3 d'
}

def cif2xyz(cif_files, output_dir):
    for cif_file in tqdm(cif_files):
        try:
            frame = read(cif_file, format='cif')
        except SpacegroupNotFoundError:
            f = open(cif_file, 'r')
            lines = f.readlines()
            f.close()
            g = TemporaryFile('w+')
    
            # Ensure correct capitalization of the H-M symbol
            for line in lines:
                if line.startswith('_symmetry_space_group_name_H-M'):
                    split_line = shlex.split(line)
                    split_line[-1] = split_line[-1].strip().capitalize()
                    split_line[-1] = spacegroup_name_change.get(
                        split_line[-1], split_line[-1]
                    )
                    split_line = ' '.join(split_line) + '\n'
                else:
                    split_line = line
    
                g.write(split_line)
    
            g.seek(0)
            frame = read(g, format='cif')
            g.close()
    
        xyz_file = os.path.splitext(os.path.basename(cif_file))[0] + '.xyz'
        write(f'{output_dir}/{xyz_file}', frame, format='extxyz')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('glob', type=str,
            help='Glob pattern for finding cif files to convert')
    parser.add_argument('output_dir', type=str, default='.',
            help='Output directory for the converted files')
    parser.add_argument('--restart', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cif_files = sorted(glob.glob(args.glob))
    if args.restart:
        cif_files_restart = []
        for cif_file in cif_files:
            basename = removesuffix(get_basename(cif_file), '.cif')
            try:
                if not os.path.getsize(f'{args.output_dir}/{basename}.xyz') > 0:
                    cif_files_restart.append(cif_file)
            except OSError:
                cif_files_restart.append(cif_file)

        cif_files = cif_files_restart

    cif2xyz(cif_files, args.output_dir)
