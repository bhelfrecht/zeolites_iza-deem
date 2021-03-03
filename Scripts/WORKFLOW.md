# Project Workflow
See README.md for a description of all scripts and notebooks

## IZA Geometries
Core + shell geometry optimizations for IZA
1.  run_gulp_geometry_iza.py
1.  optimization_summary_geometry_iza.py
1.  fix_gulp_geometry_iza.py
1.  optimization_summary_geometry_iza.py
1.  structure_summary_geometry_iza.py

## DEEM
Fixed-cell, shell-only geometry optimizations for Deem
1.  run_gulp_deem.py
1.  optimization_summary_deem.py
1.  structure_summary_deem.py

## Analysis setup
1.  compute_soaps.ipynb\*
1.  pre_analysis_checks.ipynb

## Core notebooks
1.  decomposition.ipynb\*
1.  regression_optimization.ipynb\*
1.  regression.ipynb\*
1.  svm_optimization.ipynb\*
1.  svm.ipynb\*
1.  pcovr_optimization.ipynb\*
1.  pcovr.ipynb\*
1.  gch.ipynb\*
1.  atom_resolved_density.ipynb\*

## DEEM Geometries
Core + shell geometry optimizations for Deem on GCH vertices;
to be run after all core notebooks
1.  run_gulp_geometry_deem.py
1.  optimization_summary_geometry_deem.py
1.  fix_gulp_geometry_deem.py
1.  optimization_summary_geometry_deem.py
1.  structure_summary_geometry_deem.py

## Post-analysis
1.  post_analysis_checks.ipynb

## Analysis and plotting
1.  Analysis/decomposition_analysis.ipynb
1.  Analysis/regression_analysis.ipynb
1.  Analysis/svm-pcovr_analysis.ipynb
1.  Analysis/gch_analysis.ipynb
1.  Analysis/soap_density_analysis.ipynb

\* These notebooks can be run from the command line
using `ipython -c "%run {notebook}.ipynb"`,
where `{notebook}` is the name of the notebook.
