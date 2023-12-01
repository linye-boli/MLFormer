import torch 
import numpy as np
from utils import multi_summation

def smooth_kernel_case(l):
    n = 2**l - 1
    xmin = 0
    xmax = np.pi
    h = (xmax - xmin)/(n+1)
    xh = torch.linspace(xmin, xmax, n+2)[1:-1][None][None]

    # input function
    uh = torch.sin(xh)**2

    # kernel function
    gh_X, gh_Y = torch.meshgrid(xh[0,0], xh[0,0]) # mesh grid 
    Khh = torch.cos(gh_Y - gh_X)[None][None]

    # numeric output function
    wh = multi_summation(Khh, uh, h)

    # analytic output function
    wh_gt = torch.sin(xh) * 4/3

    return uh, Khh, h, wh, wh_gt, xh 

def singular_smooth_kernel_case(l):
    n = 2**l - 1
    xmin = -1
    xmax = 1
    h = (xmax - xmin)/(n+1)
    xh = torch.linspace(xmin, xmax, n+2)[1:-1][None][None]

    # input function
    uh = 1 - xh**2

    # kernel function
    gh_X, gh_Y = torch.meshgrid(xh[0,0], xh[0,0]) # mesh grid 
    Khh = torch.log((gh_Y - gh_X).abs())
    Khh = torch.nan_to_num(Khh, neginf=-100.)[None][None]

    # numeric output function
    wh = multi_summation(Khh, uh, h)

    return uh, Khh, h, wh, xh 

def poisson_green_case(l):
    n = 2**l - 1
    xmin = 0
    xmax = 1
    h = (xmax - xmin)/(n+1)
    xh = torch.linspace(xmin, xmax, n+2)[1:-1][None][None]

    # input function
    uh = -(xh - 0.5)**2 + 0.25

    # kernel function
    gh_X, gh_Y = torch.meshgrid(xh[0,0], xh[0,0]) # mesh grid 
    Khh = (0.5 * (gh_Y + gh_X - (gh_Y - gh_X).abs()) - gh_Y * gh_X)[None][None]

    # numeric output function
    wh = multi_summation(Khh, uh, h)

    # analytic output function
    wh_gt = 1/12 * (xh**4 - 2*xh**3 + xh)

    return uh, Khh, h, wh, wh_gt, xh 
