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

def laplacianKernel(dMat, sigma):
    """
        Laplacian kernel
        
        ---Arguments---
        dMat: matrix of distances
        sigma: kernel width
    """
    return np.exp(-dMat/sigma)

def gaussianKernel(dMat, sigma):
    """
        Gaussian kernel
        
        ---Arguments---
        dMat: matrix of distances
        sigma: kernel width
    """
    return np.exp(-dMat**2/(2*sigma**2))

def MAE(predicted, observed):
    """
        Mean absolute error

        ---Arguments---
        predicted: vector of predicted property values
        observed: vector of observed (true) property values
    """
    if np.size(predicted) != np.size(observed):
        sys.exit("Predicted and observed vectors not same length")
    else:
        absErr = np.abs(predicted-observed)
        sup = np.amax(absErr)
        mae = np.sum(absErr)/np.size(predicted)
    return mae#, sup

def RMSE(predicted, observed):
    """
        Root mean square error

        ---Arguments---
        predicted: vector of predicted property values
        observed: vector of observed (true) property values
    """
    if np.size(predicted) != np.size(observed):
        sys.exit("Predicted and observed vectors not same length")
    else:
        rmse = np.sqrt(np.sum(np.power(predicted-observed, 2)) \
                /np.size(predicted))
    return rmse    

def get_random_structures(filename, nTotal, nRand, output='.'):
    """
        Creates new xyz file comprising a random
        selection of structures from an input xyz file

        ---Arguments---
        filename: input file
        nTotal: total number of structures (in input file)
        nRand: number of structures to select randomly
    """
    sys.stdout.write('Selecting random structures...\n')
    randIdxs = random.sample(range(0, nTotal), nRand)
    structCount = -1
    headerCount = 0 
    randCount = 0
    g = open('%s/randomSelection.xyz' % output, 'w')
    
    with open(filename) as f:
        for line in f:
            
            # If the first part of the line is a number,
            # this indicates a new structure
            if line[0].isdigit() == True:
                structCount += 1
                headerCount += 1
                if structCount in randIdxs:
                    randCount += 1
                    doStruct = True
                    g.write(line)
                else:
                    doStruct = False
            else:
                if doStruct == True:
                    g.write(line)
                else:
                    continue
    g.close()

def do_FPS(x, D=0, output='.'):
    """
        Farthest point sampling

        ---Arguments---
        x: input data to sample using FPS
        D: number of points to select
    """
    sys.stdout.write('Selecting FPS Points...\n')
    if D == 0:
        D = len(x)
    n = len(x)
    iy = np.zeros(D, np.int)

    # Faster evaluation of Euclidian Distance
    n2 = np.einsum("ai,ai->a", x, x)

    # Select first point at random
    iy[0] = np.random.randint(0, n)

    # Compute distances to all points
    dl = n2 + n2[iy[0]] - 2*np.dot(x, x[iy[0]])

    # Min max distances
    lmin = np.zeros(D)
    for i in range(1, D):
        iy[i] = np.argmax(dl)
        lmin[i-1] = dl[iy[i]]
        nd = n2 + n2[iy[i]] - 2*np.dot(x, x[iy[i]])
        dl = np.minimum(dl, nd)
        sys.stdout.write('Point: %d\r' % (i+1))
        sys.stdout.flush()
    sys.stdout.write('\n')
    np.savetxt('%s/FPS.idxs' % output, iy, fmt='%d')
    return iy

def quick_FPS(x, D=0, cutoff=1.0E-3, output='.'):
    """
        "Quick" Farthest Point Sampling

        ---Arguments---
        x: input data to sample using quick FPS
        D: number of points to select
        cutoff: minimum standard deviation for selection
    """
    sys.stdout.write('Computing Quick FPS...\n')

    # Select components where standard deviation is greater than the cutoff
    # (selects most diverse components
    quickFPS = np.where(np.std(x, axis=0)/np.mean(np.std(x, axis=0)) > cutoff)[0]
    if D != 0:
        quickFPS = quickFPS[0:D]
    np.savetxt('%s/quickFPS.idxs' % output, quickFPS, fmt='%d')
    sys.stdout.write('Selected %d points\n' % len(quickFPS))
    return quickFPS

def randomSelect(x, D=0, output='.'):
    """
        Select random points

        ---Arguments---
        x: input data to sample randomly
        D: number of points to select
    """
    idxs = range(0, len(x))
    np.random.shuffle(idxs)
    idxs = idxs[0:D]
    np.savetxt('%s/random.idxs' % output, idxs, fmt='%d')
    return idxs

def compute_SOAPs(al, d, idxs=None, batchSize=0, 
        prefix='SOAPs', output='.'):
    """
        Compute SOAP vectors

        ---Arguments---
        al: Quippy AtomsList or AtomsReader
        d: Quippy descriptor
        idxs: list of indices to keep from the SOAP vector
        batchSize: number of structures to include in a batch
        prefix: prefix of output file
    """
    sys.stdout.write('Computing SOAP vectors...\n')
    g = open('%s/SOAPFiles.dat' % output, 'w')
    SOAPFiles = []
    SOAPs = []
    if batchSize < 1:
        batchSize = len(al)
    N = 0
    for i, at in enumerate(al):
        at.set_cutoff(d.cutoff())
        at.calc_connect()
        SOAP = d.calc(at)['descriptor']
        if idxs is not None:
            SOAP = SOAP[:, idxs]
        SOAPs.append(SOAP)
        if ((i+1) % batchSize) == 0:
            SOAPs = np.concatenate(SOAPs)
            np.save('%s/%s-%d' % (output, prefix, N), SOAPs)
            g.write('%s/%s-%d.npy\n' % (os.path.abspath(output), prefix, N))
            SOAPs = []
            N += 1
        sys.stdout.write('Frame: %d\r' % (i+1))
        sys.stdout.flush()
    sys.stdout.write('\n')
    if len(SOAPs) > 0:
        SOAPs = np.concatenate(SOAPs)
        np.save('%s/%s-%d' % (output, prefix, N), SOAPs)
        g.write('%s/%s-%d.npy\n' % (os.path.abspath(output), prefix, N))
    g.close()

def build_iPCA(SOAPFiles, nPCA, batchSize, output='.'):
    """
        Builds PCA incrementally using SciKit Learn
        incremental PCA

        ---Arguments---
        SOAPFiles: list of files containing SOAP vectors
        nPCA: number of PCA components
        batchSize: batchSize for building the incremental PCA
    """
    sys.stdout.write('Building PCA...\n')
    PCABuilder = IncrementalPCA(n_components=nPCA, batch_size=batchSize)
    batch = []
    b = 0
    n = 0
    for idx, i in enumerate(SOAPFiles):
        SOAP = read_SOAP(i)
        PCABuilder.partial_fit(SOAP)
        sys.stdout.write('Batch: %d\r' % (idx+1))
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.write('Computing covariance...\n')
    C  = PCABuilder.get_covariance()
    sys.stdout.write('Saving covariance...\n')
    np.savetxt('%s/cov.dat' % output, C)
    sys.stdout.write('Saving mean...\n')
    np.savetxt('%s/mean.dat' % output, PCABuilder.mean_)
    sys.stdout.write('Saving eigenvectors...\n')
    np.savetxt('%s/eigenvectors.dat' % output, 
            np.transpose(PCABuilder.components_))
    return PCABuilder

def transform_PCA(W, SOAPMean, SOAPFiles, output='.'):
    """
        Transforms data according to the PCA

        ---Arguments---
        W: eigenvectors of the covariance matrix
        SOAPMean: mean of input data
        SOAPFiles: list of files containing SOAP vectors
    """
    sys.stdout.write('Transforming PCA...\n')
    g = open('%s/PCAFiles.dat' % output, 'w')
    for idx, i in enumerate(SOAPFiles):
        #ct = 1
        SOAP = read_SOAP(i)
        transformedSOAP = np.inner(SOAP-SOAPMean, W.T)
        np.save('%s/pca-%d' % (output, idx), transformedSOAP)
        g.write('%s/pca-%d.npy\n' % (os.path.abspath(output), idx))
        sys.stdout.write('Batch: %d\r' % (idx+1)) 
        sys.stdout.flush()
    sys.stdout.write('\n')
    g.close()

def reconstruct_PCA(W, SOAPMean, PCAFiles, useRawData=False, output='.'):
    """
        Reconstruct original data from PCA

        ---Arguments---
        W: eigenvectors of the covariance matrix
        SOAPMean: mean of the original data
        PCAFiles: list of files containing original data or PCA data
        useRawData: if True, reconstruct using original data; if False,
                    reconstruct using PCA data
        save: if True, save reconstructed data to file
    """
    sys.stdout.write('Reconstructing data from PCA...\n')
    g = open('%s/rSOAPFiles.dat' % output, 'w')
    for idx, i in enumerate(PCAFiles):
        PCA = read_SOAP(i)
        if useRawData is True:
            transformedPCA = np.inner(PCA, np.inner(W, W)) + SOAPMean
        else:
            transformedPCA = np.inner(PCA, W) + SOAPMean
        np.save('%s/rSOAP-%d' % (output, idx), transformedPCA)
        g.write('%s/rSOAP-%d.npy\n' % (os.path.abspath(output), idx))
        sys.stdout.write('Batch: %d\r' % (idx+1)) 
        sys.stdout.flush()
    sys.stdout.write('\n')
    g.close()

def build_repSOAPs(inputFiles, repIdxs):
    repSOAPs = []
    n = 0
    sys.stdout.write('Building representative environments...\n')
    for idx, i in enumerate(inputFiles):
        iSOAP = read_SOAP(i)
        subIdxs = np.intersect1d(repIdxs[np.where(repIdxs >= n)],
                repIdxs[np.where(repIdxs < (n+len(iSOAP)))]) - n
        n += len(iSOAP)
        repSOAPs.append(iSOAP[subIdxs])
    repSOAPs = np.concatenate(repSOAPs)
    return repSOAPs

def read_input(SOAPFile):
    """
        Pre-processes list of files containing SOAP/SOAP-PCA data

        ---Arguments---
        SOAPFile: file containing file names and file paths to the
                  files containing the SOAP data

    """
    f = open(SOAPFile, 'r')
    inputFiles = f.readlines()
    inputFiles = [i.strip() for i in inputFiles]
    f.close()
    return inputFiles

def read_SOAP(SOAPFile):
    if os.path.splitext(SOAPFile)[1] == '.npy':
        iSOAP = np.load(SOAPFile)
    else:
        iSOAP = np.loadtxt(SOAPFile)
    return iSOAP

def center_data(SOAPFile):
    inputFiles = read_input(SOAPFile)
    n = 0
    dataMean = None
    for idx, i in enumerate(inputFiles):
        iSOAP = read_SOAP(i)
        if dataMean is None:
            dataMean = np.sum(iSOAP, axis=0)
        else:
            dataMean += np.sum(iSOAP, axis=0)
        n += len(iSOAP)

    dataMean /= n

    centeredFile = '%s-centered.dat' % os.path.splitext(SOAPFile)[0]
    f = open(centeredFile, 'w')
    for idx, i in enumerate(inputFiles):
        iSOAP = read_SOAP(i)
        iSOAP -=  dataMean
        np.save('%s-centered' % os.path.splitext(i)[0], iSOAP)
        f.write('%s-centered.npy\n' % os.path.splitext(i)[0])
    f.close()

def sparse_kPCA_transform_test(inputFilesTrain, inputFilesTest, repIdxs, U,
        kernel='gaussian', zeta=1, width=1.0, nPCA=None, lowmem=True, output='.'):
    """
        Transform the kernel PCA
        FOR TESTING ONLY
    """
    train_SOAP = read_SOAP(inputFilesTrain[0])
    test_SOAP = read_SOAP(inputFilesTest[0])
    n = train_SOAP.shape[0]
    l = test_SOAP.shape[0]

    kLN = build_kernel(test_SOAP, train_SOAP,
            kernel=kernel, zeta=zeta, width=width, nc=None)
    kNN = build_kernel(train_SOAP, train_SOAP,
            kernel=kernel, zeta=zeta, width=width, nc=None)

    # Kernel centering based on: 
    # https://www.ics.uci.edu/~welling/classnotes/papers_class/Kernel-PCA.pdf
    L = np.ones((l, 1))/n
    N = np.ones((1, n))/n
    M1 = np.dot(np.sum(kLN, axis=1).reshape((l, 1)), N)
    M2 = np.dot(L, np.sum(kNN, axis=1).reshape((1, n)))
    M3 = np.dot(L, np.dot(np.sum(kNN), N))
    kLN -= M1 + M2 + M3
    np.savetxt('kpca_transform.dat', np.dot(kLN, U))

def sparse_kPCA_transform(inputFilesTrain, inputFilesTest, GMeanFile, GFiles, UFiles,
        kernel='gaussian', zeta=1, width=1.0, nPCA=None, lowmem=True, output='.'):
    """
        Project out-of-sample data onto an existing KPCA
    """

    N = 0
    kTN_sum = 0
    tSOAP = read_SOAP(inputFilesTest[0])
    T = tSOAP.shape[0]
    kTN = build_kernel_batch(inputFilesTrain, tSOAP, kernel='gaussian', 
            zeta=zeta, width=width, nc=nPCA, lowmem=lowmem, output=output)
    sys.stdout.write('Computing column sum of projection kernel...\n')
    sys.stdout.flush()
    for fdx, f in enumerate(kTN):
        kTNi = np.transpose(np.load('%s.npy' % f))
        N += kTNi.shape[1]
        if fdx == 0:
            kTN_sum = np.sum(kTNi, axis=1)
        else:
            kTN_sum += np.sum(kTNi, axis=1)
        sys.stdout.write('Batch: %d\r' % (fdx+1))
        sys.stdout.flush()

    sys.stdout.write('\n')

    kTN_sum /= N

    Gmean = np.load(GMeanFile)

    sys.stdout.write('Computing sum of training kernel...\n')
    sys.stdout.flush()
    kNN_sum = np.dot(Gmean, Gmean)
    kNN_rowsum = []

    sys.stdout.write('Computing column sum of training kernel...\n')
    sys.stdout.flush()
    for gdx, g in enumerate(GFiles):
        Gi = np.load(g)
        kNN_rowsum.append(np.dot(Gmean, Gi.T))
        sys.stdout.write('Batch: %d\r' % (gdx+1))
        sys.stdout.flush()

    sys.stdout.write('\n')

    kNN_rowsum = np.concatenate(kNN_rowsum)
        
    T1 = np.ones((T, 1))
    N1 = np.ones((1, N))
    kNN_rowsum = np.reshape(kNN_rowsum, ((1, N)))
    kTN_sum = np.reshape(kTN_sum, ((T, 1)))

    Nt = 0
    kout = open('%s/projFiles.dat' % os.path.abspath(output), 'w')
    sys.stdout.write('Centering and projecting...\n')
    projection = 0
    for fdx, f in enumerate(kTN):
        kTNi = np.transpose(np.load('%s.npy' % f))
        Ni = kTNi.shape[1]
        kTNi -= np.dot(kTN_sum, N1[:, Nt:Nt+Ni]) \
                + np.dot(T1, kNN_rowsum[:, Nt:Nt+Ni]) \
                - kNN_sum*np.dot(T1, N1[:, Nt:Nt+Ni])
        Ui = read_SOAP(UFiles[fdx])
        Ui = Ui[:, 0:nPCA]
        if fdx == 0:
            projection = np.dot(kTNi, Ui)
        else:
            projection += np.dot(kTNi, Ui)
        os.system('rm %s/k%d.npy' % (output, fdx))
        Nt += Ni
        sys.stdout.write('Batch: %d\r' % (fdx+1))
        sys.stdout.flush()

    np.save('%s/kpca_proj' % output, projection)
    kout.write('%s/kpca_proj.npy\n' % os.path.abspath(output))

    sys.stdout.write('\n')
    kout.close()

def sparse_kPCA(inputFiles, repIdxs, kernel='gaussian', zeta=1, width=1.0, 
        nPCA=None, lowmem=True, output='.'):
    """
       Build and transform the kernel PCA

       ---Arguments---
       inputFiles: list of files containing the data
       repIdxs: indices of environments to use as the representatives

       The procedure is adapted from that stated in the SAS/IML 15.1
       user manual: https://documentation.sas.com/?docsetId=imlug&docsetTarget=imlug_langref_sect226.htm&docsetVersion=15.1&locale=en
    """

    # Read inputfiles and build repSOAPs
    repSOAPs = build_repSOAPs(inputFiles, repIdxs)
    f = open('%s/KPCAFiles.dat' % output, 'w')

    # Build kNM
    kNM = build_kernel_batch(inputFiles, repSOAPs,
            kernel=kernel, zeta=zeta, width=width, nc=None, 
            lowmem=lowmem, output=output)

    # Build kMM
    kMM = build_kernel(repSOAPs, repSOAPs,
            kernel=kernel, zeta=zeta, width=width, nc=None)

    # Eigendecomposition on kMM
    w, U = np.linalg.eigh(kMM)
    w = np.flip(w, axis=0)
    U = np.flip(U, axis=1)

    # Take only positive eigenvalues
    w = w[w > 0]
    U = U[:, 0:w.size]

    W = np.diagflat(1.0/np.sqrt(w))

    if lowmem is True:
        # Compute G
        P = np.dot(U, np.dot(W, U.T))
        # np.save('P', P)
        Gmean = np.zeros(kMM.shape[0])
        n = 0
        gout = open('%s/GFiles.dat' % output, 'w')
        for idx, i in enumerate(kNM):
            sys.stdout.write('Building approx kernel, batch %d\r' % (idx+1))
            sys.stdout.flush()
            kNMi = np.load('%s.npy' % i)
            Gi = np.dot(kNMi, P)
            Gmean += np.sum(Gi, axis=0)
            n += kNMi.shape[0]
            np.save('%s/G-%d' % (output, idx), Gi)
            gout.write('%s/G-%d.npy\n' % (os.path.abspath(output), idx))
        gout.close()
        sys.stdout.write('\n')

        Gmean /= n
        np.save('%s/G-mean' % output, Gmean)

        G = np.zeros(kMM.shape)
        n = 0
        m = 0
        for idx, i in enumerate(kNM):
            sys.stdout.write('Centering approx. kernel, batch %d\r' % (idx+1))
            sys.stdout.flush()
            Gi = np.load('%s/G-%d.npy' % (output, idx))
            Gi -= Gmean
            G += np.dot(Gi.T, Gi)
        sys.stdout.write('\n')

        # Eigendecomposition on (G.T)*G
        w, V = np.linalg.eigh(G)
        # np.save('V', V) 
        w = np.flip(w, axis=0)
        V = np.flip(V, axis=1)
        W = np.diagflat(1.0/w)

        uout = open('%s/UFiles.dat' % output, 'w')

        # Approximate eigenvectors of kNN
        VW = np.dot(V, W)
        for idx, i in enumerate(kNM):
            sys.stdout.write('Building approx. eigenvectors '\
                    'and projecting, batch %d\r' % (idx+1))
            Gi = np.load('%s/G-%d.npy' % (output, idx))
            Ui = np.dot(Gi-Gmean, VW)
            
            # Retain desired number of principal components
            # and save the projections
            Ui = Ui[:, 0:nPCA]
            w = w[0:nPCA]
            np.save('%s/U-%d' % (output, idx), Ui) 

            Gi = np.dot(Ui, np.diagflat(w))
            np.save('%s/kpca-%d' % (output, idx), Gi)
            f.write('%s/kpca-%d.npy\n' % (os.path.abspath(output), idx))
            uout.write('%s/U-%d.npy\n' % (os.path.abspath(output), idx))
            os.system('rm %s/k%d.npy' % (output, idx))
        uout.close()
        sys.stdout.write('\n')

    else:
        pass
        # Compute G
        sys.stdout.write('Building approx. kernel...\n')
        G = np.dot(kNM, np.dot(U, np.dot(W, U.T)))

        # Center G
        sys.stdout.write('Centering approx. kernel...\n')
        G -= np.mean(G, axis=0)

        # Eigendecomposition on (G.T)*G
        w, V = np.linalg.eigh(np.dot(G.T, G))
        w = np.flip(w, axis=0)
        V = np.flip(V, axis=1)
        W = np.diagflat(1.0/w)

        # Approximate eigenvectors of kNN
        sys.stdout.write('Building approx. eigenvectors...\n')
        U = np.dot(G, np.dot(V, W))

        # Retain desired number of principal components
        U = U[:, 0:nPCA]
        w = w[0:nPCA]

        # Projection
        sys.stdout.write('Projecting...\n')
        np.save('%s/U' % output, U)
        np.save('%s/kpca-0' % output, np.dot(U, np.diagflat(w)))
        f.write('%s/kpca-0.npy\n' % os.path.abspath(output))

    f.close()

def npy_convert(fileList):
    """
        Converts from list of .npy files to ASCII

        ---Arguments---
        fileList: list of filenames to convert

    """
    for idx, i in enumerate(fileList):
        sys.stdout.write('Converting file: %d\r' % (idx+1))
        filename = os.path.splitext(i)[0]
        np.savetxt('%s.dat' % filename, np.load(i))
    sys.stdout.write('\n')

def npy_stdout(fileList):
    """
        Reads .npy file and prints to stdout

        ---Arguments---
        fileList: list of filenames to convert
    """
    n = 0
    for i in fileList:
        data = np.load(i)
        n += len(data)
        for j in data:
            print '    '.join(map(str, j))

def extract_structure_properties(al, Z, propName=None):
    """
        Extracts structue properties from xyz file

        ---Arguments---
        al: Quippy AtomsList or AtomsReader
        Z: central atom species
        propName: name of property to extract
    """
    p = np.zeros(len(al))
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
        if propName != 'volume':
            p[i] = at.params[propName]
        else:
            p[i] = volume[i]

    return structIdxs, nAtoms, volume, p

def build_kernel_batch(inputFiles, SOAPs2, kernel='linear', 
        zeta=1, width=1.0, nc=None, lowmem=False, output='.'):
    """
        SOAP kernel between two SOAP vectors in batches

        ---Arguments---
        inputFiles: list of filenames containing
                    SOAP vectors
        SOAPs2: input SOAP vectors
        zeta: exponent for nonlinear kernel
    """
    sys.stdout.write('Building kernel...\n')
    kList = []
    for idx, i in enumerate(inputFiles):
        SOAPs1 = read_SOAP(i)[:, 0:nc]
        if kernel == 'gaussian':
            d = cdist(SOAPs1, SOAPs2, metric='euclidean')
            k = gaussianKernel(d, width)
        elif kernel == 'laplacian':
            d = cdist(SOAPs1, SOAPs2, metric='cityblock')
            k = laplacianKernel(d, width)
        else:
            k = np.dot(SOAPs1, SOAPs2.T)**zeta
        if lowmem is True:
            np.save('%s/k%d' % (output, idx), k)
            kList.append('%s/k%d' % (output, idx))
        else:
            kList.append(k)
        sys.stdout.write('Batch: %d\r' % (idx+1))
        sys.stdout.flush()
    if lowmem is True:
        k = kList
    else:
        k = np.concatenate(kList)
    sys.stdout.write('\n')
    return k

def build_kernel(SOAPs1, SOAPs2, kernel='linear', zeta=1, width=1.0, nc=None):
    """
        SOAP kernel between two SOAP vectors

        ---Arguments---
        SOAPs1, SOAPs2: input SOAP vectors
        zeta: exponent for nonlinear kernel
    """
    sys.stdout.write('Building kernel...\n')
    SOAPs1 = SOAPs1[:, 0:nc]
    SOAPs2 = SOAPs2[:, 0:nc]
    if kernel == 'gaussian':
        d = cdist(SOAPs1, SOAPs2, metric='euclidean')
        k = gaussianKernel(d, width)
    elif kernel == 'laplacian':
        d = cdist(SOAPs1, SOAPs2, metric='cityblock')
        k = laplacianKernel(d, width)
    else:
        k = np.dot(SOAPs1, SOAPs2.T)**zeta
    return k

def build_sum_kernel_batch(inputFiles, SOAPs2, structIdxs, kernel='linear', 
        zeta=1, width=1.0, nc=None):
    """
        Build sum kernel for a structure

        ---Arguments---
        inputFiles: list of filenames of files containing SOAP vectors 
        SOAPs2: input SOAP vectors
        structIdxs: list of indices indicating which
                    SOAP vectors belong to which structure
                    (output by extract_structure_properties)
        zeta: exponent for nonlinear kernel
        width: width for Gaussian or Laplacian kernel
        kernel: type of kernel to build
        nc: number of components to use
    """
    sys.stdout.write('Building sum kernel...\n')
    k = np.zeros((len(structIdxs), len(SOAPs2)))
    n = 0
    for i in inputFiles:
        SOAPs1 = read_SOAP(i)[:, 0:nc]
        m = 0
        while m < len(SOAPs1):
            iSOAPs1 = SOAPs1[m:m+structIdxs[n]]
            if kernel == 'gaussian':
                dj = cdist(iSOAPs1, SOAPs2, metric='euclidean')
                kj = gaussianKernel(dj, width)
                k[n, :] = np.sum(kj, axis=0)
            elif kernel == 'laplacian':
                dj = cdist(iSOAPs1, SOAPs2, metric='cityblock')
                kj = laplacianKernel(dj, width)
                k[n, :] = np.sum(kj, axis=0)
            else:
                k[n, :] = np.sum(np.dot(iSOAPs1, SOAPs2.T)**zeta, axis=0)
            m += structIdxs[n]
            n += 1
    return k

def build_sum_kernel(SOAPs1, SOAPs2, structIdxs, kernel='linear', 
        zeta=1, width=1.0):
    """
        Build sum kernel for a structure

        ---Arguments---
        SOAPs1, SOAPs2: input SOAP vectors
        structIdxs: list of indices indicating which
                    SOAP vectors belong to which structure
                    (output by extract_structure_properties)
        zeta: exponent for nonlinear kernel
    """
    sys.stdout.write('Building sum kernel...\n')
    k = np.zeros((len(structIdxs), len(SOAPs2)))
    n = 0
    for i in range(0, len(structIdxs)):
        iSOAP = SOAPs1[n:structIdxs[i]+n]
        if kernel == 'gaussian':
            dj = cdist(iSOAP, SOAPs2, metric='euclidean')
            kj = gaussianKernel(dj, width)
            k[i, :] = np.sum(kj, axis=0)
        elif kernel == 'laplacian':
            dj = cdist(iSOAP, SOAPs2, metric='euclidean')
            kj = gaussianKernel(dj, width)
            k[i, :] = np.sum(kj, axis=0)
        else:
            k[i, :] = np.sum(np.dot(iSOAP, SOAPs2.T)**zeta, axis=0)
        n += structIdxs[i]
    return k

def property_regression(y, kMM, kNM, nStruct, idxsTrain, idxsValidate, 
                        sigma=1.0, jitter=1.0E-16, 
                        envKernel=None, output='.'):
    """
        Perform property decomposition

        ---Arguments---
        y: structural property data
        kMM: kernel between representative environments
        kNM: sum kernel of representative environments
        SOAPs1, SOAPs2: full input SOAP vectors
        nStruct: number of structures
        idxsTrain: training indices
        idxsValidate: validation (or testing) indices
        sigma: regularization parameter
        jitter: value for additional regularization
        save: save regression data
        envKernel: compute environment decomposition
    """

    # Solve KRR problem
    delta = np.var(y)*len(kMM)/np.trace(kMM)
    K = kMM*delta*sigma**2 + np.dot(kNM[idxsTrain].T, kNM[idxsTrain])*delta**2
    maxEigVal = np.amax(np.linalg.eigvalsh(K))
    K += np.eye(len(kMM))*maxEigVal*jitter
    Y = delta*np.dot(delta*kNM[idxsTrain].T, y[idxsTrain])
    w = np.linalg.solve(K, Y)
    
    # Predict structure properties
    yy = np.dot(kNM, w)
    
    # Build environment kernel
    # need to use the build_kernel function here
    if envKernel is not None:
        if isinstance(envKernel, list):
            for idx, i in enumerate(envKernel):
                iKernel = read_SOAP('%s.npy' % i)
                yyEnv = np.dot(iKernel, w)
                np.savetxt('%s/envProperties-%d.dat' % (output, idx), yyEnv)
        else:

            # Decompose structural property into 
            # environmental contributions; save
            yyEnv = np.dot(envKernel, w)
            np.savetxt('%s/envProperties.dat' % output, yyEnv)

    return y[idxsTrain], y[idxsValidate], yy[idxsTrain], yy[idxsValidate], w

def property_regression_oos(w, k, output='.'):
    if isinstance(k, list):
        for idx, i in enumerate(k):
            ik = read_SOAP('%s.npy' % i)
            y = np.dot(ik, w)
            np.savetxt('%s/envProperties_proj-%d.dat' % (output, idx), y)
            #os.system('rm %s.npy' % i)
    else:
        y = np.dot(ik, w)
        np.savetxt('%s/envProperties_proj.dat' % output, y)

def kernel_distance(ii, jj, kij):
    """
        Compute kernel induced distance

        ---Arguments---
        ii: Diagonal of K(A, A)
        jj: Diagonal of K(B, B)
        kij: K(A, B)
    """
    
    # ii and jj are diagonals of kii and kjj kernels
    # kij is kernel between i and j
    radicand = -2.0*kij + np.reshape(ii, (len(ii), 1)) + jj

    # Handle machine precision errors around 0
    radicand[np.where(radicand < 0.0)] = 0.0
    D = np.sqrt(radicand)
    return D

def kernel_histogram_rectangular(D, bins=200):
    """
        Compute histogram of kernel-induced distances
        for rectangular kernel

        ---Arguments---
        D: matrix of distances
        bins: number of histogram bins
    """
    H, binEdges = np.histogram(D.flatten(), bins=bins, density=True)
    return H, binEdges

def kernel_histogram_square(D, bins=200):
    """
        Compute histogram of kernel-induced distances
        for square kernel

        ---Arguments---
        D: matrix of distances
        bins: number of histogram bins
    """
    D = D[np.triu_indices(len(D))]
    H, binEdges = np.histogram(D, bins=bins, density=True)
    return H, binEdges

def kernel_histogram_min(D, bins=200, axis=None):
    """
        Compute histogram of minimum kernel-induced distances

        ---Arguments---
        D: matrix of distances
        bins: number of histogram bins
        axis: axis over which to minimize
    """
    D = np.amin(D, axis=axis)
    H, binEdges = np.histogram(D, bins=bins, density=True)
    return H, binEdges

