import os
import scipy
import torch 
import numpy as np
from einops import rearrange
from utils import multi_summation, numeric_integ

def reference_test(l, kernel='cosine'):
    # numerical kernel evaulation 
    Khh, xh, h = numerical_kernel(l, kernel)

    # input function
    if kernel == 'cosine':
        uh = torch.sin(xh)**2
    elif kernel == 'lnabs':
        uh = 1 - xh**2 + torch.sin(xh)**2 +100 + torch.exp(xh)

    elif kernel == 'laplace':
        uh = -(xh - 0.5)**2
    else:
        uh = 1 - xh**2

    # numeric output function
    wh = multi_summation(Khh, uh, h)
    # wh = numeric_integ(Khh, uh, h)

    if kernel == 'cosine':
        # analytic output function
        wh_gt = torch.sin(xh) * 4/3
    elif kernel == 'laplace':
        wh_gt = 1/12 * (xh**4 - 2*xh**3 + xh)
    else:
        wh_gt = None
    
    return uh, Khh, h, wh, wh_gt, xh 


def build_mesh1d(xmin, xmax, n):
    h = (xmax - xmin)/(n+1)
    xh = torch.linspace(xmin, xmax, n+2)
    gh_X, gh_Y = torch.meshgrid(xh, xh, indexing="ij") # mesh grid 
    return h, xh, gh_X, gh_Y

def numerical_kernel(l, kernel_type='cosine'):
    n = 2**l - 1

    if kernel_type == 'cosine':
        h, xh, gh_X, gh_Y = build_mesh1d(0, np.pi, n)
        Khh = torch.cos(gh_Y - gh_X)
    elif kernel_type == 'lnabs':
        h, xh, gh_X, gh_Y = build_mesh1d(-1, 1, n)
        Khh = torch.log((gh_Y - gh_X).abs())
        Khh = torch.nan_to_num(Khh, neginf=-1e2)
    elif kernel_type == 'laplace':
        h, xh, x, y = build_mesh1d(0, 1, n)
        Khh = x*(1-y)*(x<=y) + y*(1-x)*(x>y)
    elif kernel_type == 'advection_diffusion':
        h, xh, x, y = build_mesh1d(0, 1, n)
        Khh = 4*np.exp(-2*(x-y))*((y-1)*x*(x<=y)+(x-1)*y*(x>y))
    elif kernel_type == 'helmholtz':
        h, xh, x, y = build_mesh1d(0, 1, n)
        Khh = (15*np.sin(15))**(-1)*np.sin(15*x)*np.sin(15*(y-1))*(x<=y)+(15*np.sin(15))**(-1)*np.sin(15*y)*np.sin(15*(x-1))*(x>y)
    elif kernel_type == 'negative_helmholtz':
        h, xh, x, y = build_mesh1d(0, 1, n)
        Khh = (8*np.sinh(8))**(-1)*np.sinh(8*x)*np.sinh(8*(y-1))*(x<=y) + (8*np.sinh(8))**(-1)*np.sinh(8*y)*np.sinh(8*(x-1))*(x>y)

    return Khh[None][None], xh[None][None], h

def load_dataset_1d(kernel, data_root, ntrain=1000, ntest=200, bsz=64, odd=True):

    data_path = os.path.join(data_root, kernel+'.mat')
    raw_data = scipy.io.loadmat(data_path)

    if odd:
        F = raw_data['F']
        U = raw_data['U']
        U_hom = raw_data['U_hom']
    else:
        F = raw_data['F'][1:]
        U = raw_data['U'][1:]
        U_hom = raw_data['U_hom'][1:]
    
    if U_hom.sum() == 0:
        F = F / U.max()
        U = U / U.max()
        
    us = rearrange(F, 'n b -> b 1 n')
    ws = rearrange(U, 'n b-> b 1 n')

    us = torch.tensor(us).float()
    ws = torch.tensor(ws).float()

    us_train = us[:ntrain]
    us_test = us[-ntest:]
    ws_train = ws[:ntrain]
    ws_test = ws[-ntest:]
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(us_train, ws_train), batch_size=bsz, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(us_test, ws_test), batch_size=bsz, shuffle=False)

    xh, yh = raw_data['X'], raw_data['Y']
    x, y = np.meshgrid(xh, yh)
    grid_pts = np.concatenate([[x], [y]])
    grid_pts = rearrange(grid_pts, 'c m n -> m n c')
    grid_pts = torch.tensor(grid_pts).float()
    h = 2**(-13)

    if 'ExactGreen' in raw_data.keys():
        Khh = torch.tensor(eval(raw_data['ExactGreen'][0]).T).float()[None,None]
    else:
        Khh = None

    return train_loader, test_loader, Khh, U_hom, xh, grid_pts, h