#!/usr/bin/python

import os
import sys
import argparse
import random
import quippy as qp
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import scale
from sklearn import svm

def correlation_factor(PCAFiles, al, Z, nPCA):
    structIdxs = []
    for i, at in enumerate(al):
        n = 0
        atoms = at.get_atomic_numbers()
        for j in Z:
            n += np.count_nonzero(atoms == j)
        structIdxs.append(n)
    
    f = 0
    s = 0
    ss = 0
    CF = []
    if os.path.splitext(PCAFiles[f])[1] == '.npy':
        batchPCA = np.load(PCAFiles[f])[:, 0:nPCA]
    else:
        batchPCA = np.loadtxt(PCAFiles[f])[:, 0:nPCA]

    for i in range(0, len(structIdxs)):
        iPCA = batchPCA[s:s+structIdxs[i]]
        iMean = np.mean(iPCA, axis=0)
        iPCA -= iMean
        iKernel = np.dot(iPCA, iPCA.T)
        #CF.append(np.sum(np.triu(iKernel, k=1))/structIdxs[i])
        CF.append(np.sum(np.triu(iKernel, k=0))/structIdxs[i])
        s += structIdxs[i]
        ss += 1
        if s >= len(batchPCA) and (f+1) < len(PCAFiles):
            f += 1
            if os.path.splitext(PCAFiles[f])[1] == '.npy':
                batchPCA = np.load(PCAFiles[f])[:, 0:nPCA]
            else:
                batchPCA = np.loadtxt(PCAFiles[f])[:, 0:nPCA]
            s = 0
        sys.stdout.write('Batch: %d, Structure: %d\r' % (f+1, ss))
    sys.stdout.write('\n')
    CF = np.asarray(CF)
    np.savetxt('CF.dat', CF)

def DEEM_score(DEEMFiles, IZAFiles, al, Z, nc=None, propName=None, m='euclidean'):
    if propName is not None:
        p = np.zeros(len(al))
    else:
        p = None
    nAtoms = np.zeros(len(al))
    volume = np.zeros(len(al))
    structIdxs = []
    for i, at in enumerate(al):
        n = 0
        atoms = at.get_atomic_numbers()
        for j in Z:
            n += np.count_nonzero(atoms == j)
        structIdxs.append(n)
        nAtoms[i] = len(atoms)
        volume[i] = np.linalg.det(at.cell)
        if propName is not None:
            p[i] = at.params[propName]

    volumes = np.repeat(volume, structIdxs)
    if propName is not None:
        ps = np.repeat(p, structIdxs)
    
    f = 0
    s = 0
    ss = 0
    score = []
    envscore = []
    mins = []
    envmin = []
    if os.path.splitext(DEEMFiles[f])[1] == '.npy':
        batchDEEM = np.load(DEEMFiles[f])[:, 0:nc]
    else:
        batchDEEM = np.loadtxt(DEEMFiles[f])[:, 0:nc]
    if os.path.splitext(IZAFiles[f])[1] == '.npy':
        batchIZA = np.load(IZAFiles[f])[:, 0:nc]
    else:
        batchIZA = np.loadtxt(IZAFiles[f])[:, 0:nc]

    for i in range(0, len(structIdxs)):
        iDEEM = batchDEEM[s:s+structIdxs[i]]
        dDEEM = cdist(iDEEM, batchIZA, metric=m)
        dMin = np.amin(dDEEM, axis=1)
        dMaxMin = np.amax(dMin)
        score.append(dMaxMin)
        envscore.append([dMaxMin]*structIdxs[i])
        mins.append(np.amin(dMin))
        envmin.append(dMin)
        s += structIdxs[i]
        ss += 1
        if s >= len(batchDEEM) and (f+1) < len(DEEMFiles):
            f += 1
            if os.path.splitext(DEEMFiles[f])[1] == '.npy':
                batchDEEM = np.load(DEEMFiles[f])[:, 0:nc]
            else:
                batchDEEM = np.loadtxt(DEEMFiles[f])[:, 0:nc]
            s = 0
        sys.stdout.write('Batch: %d, Structure: %d\r' % (f+1, ss))
    sys.stdout.write('\n')
    score = np.asarray(score)
    envscore = np.concatenate(envscore)
    mins = np.asarray(mins)
    envmin = np.concatenate(envmin)
    np.savetxt('maxmin.dat', score)
    np.savetxt('maxmins.dat', envscore)
    np.savetxt('mins.dat', mins)
    np.savetxt('envmin.dat', envmin)
    np.savetxt('volume.dat', volume)
    np.savetxt('volumes.dat', volumes)
    if propName is not None:
        np.savetxt('p.dat', p)
        np.savetxt('ps.dat', ps)

def vDist(v1, v2, kType):
    """
        Compute Euclidian or Manhattan norm
        between two vectors
        
        ---Arguments---
        v1, v2: vectors
        kType: norm type
    """
    if np.shape(v1) != np.shape(v2):
        sys.exit("Vectors are not same length")
    else:
        if kType == 'gaussian':
            return np.linalg.norm((v1-v2), ord=2)
        else:
            return np.linalg.norm((v1-v2), ord=1)

def matDist(mat1, mat2, kType):
    """
        Compute Euclidian or Manhattan norm
        between two matrices
        
        ---Arguments---
        mat1, mat2: matrices
        kType: norm type
    """
    if np.shape(mat1) != np.shape(mat2):
        sys.exit("Feature Matrices are not the same shape")
    else:
        if kType == 'gaussian':
            return np.linalg.norm((mat1-mat2).flatten(), ord=2)
        else:
            return np.linalg.norm((mat1-mat2).flatten(), ord=1)

def build_covariance(SOAPFiles):
    """
        Iteratively builds covariance

        ---Arguments---
        SOAPFiles: list of files containing SOAP vectors in ASCII format
    """
    sys.stdout.write('Building covariance...\n')
    n = 0
    for i in SOAPFiles:
        with open(i, 'r') as f:
            for line in f:
                SOAP = np.asarray([float(x) for x in line.strip().split()])
                if n == 0:
                    p = np.shape(SOAP)[0]
                    SOAPMean = np.zeros(p)
                    C = np.zeros((p, p))
                n += 1
                C += (n-1)/float(n) * np.outer(SOAP-SOAPMean, SOAP-SOAPMean)
                SOAPMean = ((n-1)*SOAPMean + SOAP)/n 
                sys.stdout.write('Center: %d\r' % n)
    sys.stdout.write('\n')
    C = np.divide(C, n-1)
    sys.stdout.write('Saving covariance...\n')
    np.savetxt('cov.dat', C)
    sys.stdout.write('Saving mean...\n')
    np.savetxt('mean.dat', SOAPMean)

def build_PCA(C, nPCA):
    """
        Builds PCA from an input covariance matrix
        
        ---Arguments---
        C: covariance matrix
        nPCA: number of PCA components
    """
    sys.stdout.write('Building PCA...\n')
    p = np.shape(C)[0]
    u, V = np.linalg.eigh(C)
    u = np.flip(u, axis=0)
    V = np.flip(V, axis=1)
    D = np.zeros((p, p))
    g = np.zeros(p)
    for i in range(0, p):
        D[i, i] = u[i]
        g[i] = np.sum(D[0:i+1, 0:i+1])

    varRatio = g[0:nPCA]/g[-1]
    print "Variance Ratio", varRatio
    W = V[:, 0:nPCA]
    np.savetxt('eigenvectors.dat', W)
    np.savetxt('ratio.dat', varRatio)
    return W

def build_structure_kernel(SOAPs1, SOAPs2, structIdxs, zeta=1):
    """
        Build structural kernel

        ---Arguments---
        SOAPs1, SOAPs2: input SOAP vectors
        structIdxs: list of indices indicating which
                    SOAP vectors belong to which structure
                    (output by extract_structure_properties)
        zeta: exponent for nonlinear kernel
    """
    k = np.zeros((len(structIdxs), len(structIdxs)))
    n = 0
    for i in range(0, len(structIdxs)):
        m = 0
        iSOAP = SOAPs1[n:strucIdxs[i]+n]
        kRow = np.sum(np.dot(iSOAP, SOAPs2.T)**zeta, axis=0)
        for j in range(0, len(structIdxs)):
            k[i, j] = np.sum(kRow[m:structIdxs[j]+m], axis=1)
            m += structIdxs[j]
        n += structIdxs[i]
    return k
