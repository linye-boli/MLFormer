import sys 
sys.path.append('../')

from tqdm import tqdm
import scipy
import numpy as np
from dataset import numerical_kernel

l = 13
us = scipy.io.loadmat(
    '/workdir/MLFormer/data/kernel_estimation/grf1d_N2000_s8193_m0_gamma1_tau1_sigma1_dirichlet.mat')['f'][:,1:-1]
kernel_types = ['cosine', 'lnabs', 'laplace', 'advection_diffusion', 'helmholtz', 'negative_helmholtz']

for kernel in tqdm(kernel_types, total=len(kernel_types)):
    Khh, xh, h = numerical_kernel(l, kernel_type=kernel)
    ws = np.einsum('mn,bn->bm', Khh[0,0].numpy(), us) * h
    np.save('/workdir/MLFormer/data/kernel_estimation/{:}_8191.npy'.format(kernel), ws)