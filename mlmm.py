import pdb 
import torch 
import time
import torch.nn.functional as F
from einops import repeat, rearrange
from utils import (
    restrict1d, 
    interp1d, interp1d_cols, interp1d_rows, 
    multi_summation, 
    injection1d, injection1d_cols, injection1d_rows)
import torch.nn as nn

def fetch_nbrs(n, m):
    # n : int, lenth of inputs,
    # m : int, radius of window
 
    idx_h = torch.arange(n)
    idx_nbrs = torch.arange(-m, m+1)
    idx_j = torch.cartesian_prod(idx_h, idx_nbrs).sum(axis=1).reshape(-1, 2*m+1) # n x 2m+1
    idx_i = repeat(idx_h, 'i -> i m', m=2*m+1)

    return idx_i, idx_j

def SmoothKernelMLMM_full(uh, Khh, h, k=3, order=2):

    for i in range(k):
        uh = restrict1d(uh, order)
        # coarsen Khh by injection and uh by restriction
        Khh = injection1d_rows(injection1d_cols(Khh))
        h = h*2
            
    # coarsest approximation
    wh = multi_summation(Khh, uh, h)
    wh_ml = [wh]

    for i in range(k):
        wh = interp1d(wh, order)
        wh_ml.append(wh)
    
    return wh_ml, Khh

def SmoothKernelMLMM(KHH, uh, h, k):
    for i in range(k):
        uh = restrict1d(uh)
        h = h*2

    wh = multi_summation(KHH, uh, h)
    wh_ml = [wh]

    for i in range(k):
        wh = interp1d(wh)
        wh_ml.append(wh)

    return wh_ml

def SingularSmoothKernelMLMM(KHH, uh, Khh_banddiff_lst, boundary_lst, idx_j_lst, h, k, order=2):
    assert len(idx_j_lst) == k
    uh_band_lst = []

    for i in range(k):
        n = uh.shape[-1]
        idx_mask = (idx_j_lst[i] >= 0) & (idx_j_lst[i] <= n-1)
        uh_band = uh[:,:,idx_j_lst[i]] * idx_mask
        uh_band_even = uh_band[:,:,::2]
        uh_band_odd = uh_band[:,:,1::2]
        uh_band_lst.append([uh_band_even[:,:,:,::2], uh_band_odd])
        uh = restrict1d(uh, order=order)
        # update       
        h = h*2

    # coarsest approximation
    wh = multi_summation(KHH, uh, h)
    wh_ml = [wh]

    # reverse list
    uh_band_lst = uh_band_lst[::-1]
    
    # multi-level correction
    for i in range(k):
        h = h/2
        wh_even_corr = (Khh_banddiff_lst[i][0]*uh_band_lst[i][0]).sum(axis=-1)*h
        wh_even = wh 
        wh[:,:,[0]] = boundary_lst[0][i]
        wh[:,:,[-1]] = boundary_lst[1][i]
        wh_even[:,:,1:-1] += wh_even_corr[:,:,1:-1]
        wh = interp1d(wh_even, order=order)
        wh[:,:,1::2] = wh[:,:,1::2] + (Khh_banddiff_lst[i][1] * uh_band_lst[i][1]).sum(axis=-1)*h

        wh_ml.append(wh)

    return wh_ml

def SingularSmoothKernelMLMM_full(uh, Khh, h, k=3, order=2):

    corr_lst = []
    uh_lst = []
    lb_lst = []
    rb_lst = []

    for i in range(k):
        wh = multi_summation(Khh, uh, h)
        lb_lst.append(wh[:,:,[0]])
        rb_lst.append(wh[:,:,[-1]])

        KHh = injection1d_rows(Khh) 
        KHH = injection1d_cols(KHh)
        KHh_smooth = interp1d_cols(KHH, order=order) # \tilde{K}
        KHh_corr_even = KHh - KHh_smooth

        Khh_smooth = interp1d_rows(KHh, order=order)[:,:,1::2]
        Khh_corr_odd = Khh[:,:,1::2] - Khh_smooth

        corr_lst.append([KHh_corr_even, Khh_corr_odd])
        uh_lst.append(uh)
        
        uh = restrict1d(uh, order=order)
        h = h*2
        Khh = KHH
    
    corr_lst = corr_lst[::-1]
    uh_lst = uh_lst[::-1]
    lb_lst = lb_lst[::-1]
    rb_lst = rb_lst[::-1]

    wh = multi_summation(Khh, uh, h)
    wh_ml = [wh]
    for i in range(k):
        h = h / 2
        wh_corr = multi_summation(corr_lst[i][0], uh_lst[i], h)
        wh += wh_corr
        # wh[:,:,1:-1] += wh_corr[:,:,1:-1]
        wh[:,:,[0]] = lb_lst[i]
        wh[:,:,[-1]] = rb_lst[i]
        wh = interp1d(wh, order=order)
        wh[:,:,1::2] = wh[:,:,1::2] + multi_summation(corr_lst[i][1], uh_lst[i], h)
        # wh = wh_
        wh_ml.append(wh)

    return wh_ml, Khh
    
def SingularSmoothKernelMLMM_local(uh, Khh, h, k=3, order=2, m=7):
    Khh_banddiff_lst = []
    uh_band_lst = []
    lb_lst = []
    rb_lst = []
    idx_j_lst = []

    for i in range(k):
        # calculate boundary values
        w_lb = multi_summation(Khh[:,:,[0]], uh, h)
        w_rb = multi_summation(Khh[:,:,[-1]], uh, h)
        lb_lst.append(w_lb)
        rb_lst.append(w_rb)

        # evaluate kernel function on coarse grid
        KHh = injection1d_rows(Khh) 
        KHH = injection1d_cols(KHh)

        # smooth approximation of kernel function
        KHh_smooth = interp1d_cols(injection1d_cols(Khh), order=order)
        Khh_smooth = interp1d_rows(KHh, order=order)

        # fetch nbr idx
        n = Khh.shape[-1] 
        idx_i, idx_j = fetch_nbrs(n, m)
        idx_mask = (idx_j >= 0) & (idx_j <= n-1)
        idx_j[idx_j < 0] = 0 
        idx_j[idx_j > n-1] = n-1
        idx_j_lst.append(idx_j)

        # band diff between fine kernel and smooth approximation
        KHh_banddiff_even = (Khh - KHh_smooth)[:,:,idx_i, idx_j][:,:,::2]
        Khh_banddiff_odd =  (Khh - Khh_smooth)[:,:,idx_i, idx_j][:,:,1::2]

        # correction kernels
        Khh_banddiff_lst.append(
            [KHh_banddiff_even[:,:,:,::2], Khh_banddiff_odd])

        # uh band
        uh_band = uh[:,:,idx_j] * idx_mask
        uh_band_even = uh_band[:,:,::2]
        uh_band_odd = uh_band[:,:,1::2]
        uh_band_lst.append([uh_band_even[:,:,:,::2], uh_band_odd])

        # coarse uh
        uh = restrict1d(uh, order=order)

        # update       
        h = h*2
        Khh = KHH
    
    # reverse list
    Khh_banddiff_lst = Khh_banddiff_lst[::-1]
    uh_band_lst = uh_band_lst[::-1]
    boundary_lst = [lb_lst[::-1], rb_lst[::-1]]

    # coarsest approximation
    wh = multi_summation(Khh, uh, h)    
    wh_ml = [wh]

    # multi-level correction
    for i in range(k):
        h = h/2        
        wh_even_corr = (Khh_banddiff_lst[i][0]*uh_band_lst[i][0]).sum(axis=-1)*h
        wh += wh_even_corr
        wh[:,:,[0]] = boundary_lst[0][i]
        wh[:,:,[-1]] = boundary_lst[1][i]
        wh = interp1d(wh, order=order)
        wh[:,:,1::2] = wh[:,:,1::2] + (Khh_banddiff_lst[i][1] * uh_band_lst[i][1]).sum(axis=-1)*h
        wh_ml.append(wh)
    
    return wh_ml, Khh, Khh_banddiff_lst, boundary_lst, idx_j_lst

def SmoothKernelReconstruction(KHH, l, k):
    Khh = KHH
    Khh_lst = [Khh]
    for i in range(k):
        Khh = interp1d_rows(interp1d_cols(Khh))
        Khh_lst.append(Khh)
    assert Khh.shape[-1] == 2**l+1
    return Khh_lst

def SingularSmoothKernelReconstruction(KHH, Khh_banddiff_lst, l, k, m):
    idx_i_lst = []
    idx_j_lst = []

    for i in range(k):
        n = 2**(l-i)+1
        idx_i, idx_j = fetch_nbrs(n, m) 
        idx_j[idx_j < 0] = 0 
        idx_j[idx_j > n-1] = n-1 
        idx_i_lst.append(idx_i)
        idx_j_lst.append(idx_j)
    
    idx_i_lst = idx_i_lst[::-1]
    idx_j_lst = idx_j_lst[::-1]
    Khh = KHH
    Khh_lst = [Khh]

    for i in range(k):
        Khh = interp1d_rows(interp1d_cols(Khh))
        Khh[:,:,idx_i_lst[i][::2,::2], idx_j_lst[i][::2,::2]] += Khh_banddiff_lst[i][0]
        Khh = interp1d_rows(Khh[:,:,::2])
        Khh[:,:,idx_i_lst[i][1::2], idx_j_lst[i][1::2]] += Khh_banddiff_lst[i][1]
        Khh_lst.append(Khh)

    return Khh_lst

if __name__ == '__main__':

    from dataset import reference_test

    # ------------------------------------------
    l = 8 # number of level, total number of points is 2^l-1
    k = 1 # number of corase level
    m = 7 # local range for correction
    order = 2 # order of interpolation/restriction

    uh, Khh, h, wh_numeric, wh_analytic, xh = reference_test(l, 'lnabs')
    wh_singular_ml, KHH, Khh_banddiff_lst, idx_j_lst = SingularSmoothKernelMLMM_local(uh, Khh, h, k, order, m)
    Khh_singular = SingularSmoothKernelReconstruction(KHH, Khh_banddiff_lst, l, k, m)
    