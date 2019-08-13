#!/usr/bin/python

import os
import sys
import numpy as np
from sklearn.decomposition import KernelPCA
import SOAPTools

inputFiles = SOAPTools.read_input(sys.argv[1])
testFiles = SOAPTools.read_input(sys.argv[2])

soap = SOAPTools.read_SOAP(inputFiles[0])
#randIdxs = np.arange(0, soap.shape[0])
#np.random.shuffle(randIdxs)
#soap = soap[randIdxs[0:10000], :]
soap_test = SOAPTools.read_SOAP(testFiles[0])

kpca = KernelPCA(n_components=None, kernel='rbf', gamma=1.0/(2*0.3**2), copy_X=False)
#kpca = KernelPCA(n_components=None, kernel='linear', gamma=1.0/(2*0.3**2), copy_X=False)
transformed_soap = kpca.fit_transform(soap)
transformed_soap_test = kpca.transform(soap_test)

np.savetxt('kpc_skl.dat', transformed_soap)
np.savetxt('kpc_skl_test.dat', transformed_soap_test)
