import pdb 
import torch 
import time
import torch.nn.functional as F
from einops import repeat
from utils import restrict1d, interp1d, interp1d_cols, interp1d_rows, multi_summation

def fetch_nbrs(n, m):
    # n : int, lenth of inputs,
    # m : int, radius of window
 
    idx_h = torch.arange(1, n+1)
    idx_nbrs = torch.arange(-m, m+1)
    idx_j = torch.cartesian_prod(idx_h, idx_nbrs).sum(axis=1).reshape(-1, 2*m+1) # n x 2m+1
    idx_i = repeat(idx_h, 'i -> i m', m=2*m+1)

    return idx_i, idx_j

def local_correction(wH, Khh_correction, uh_chunk, h, order=2):
    # even row local correction
    wH = wH + (Khh_correction[:,:,1::2,::2] * uh_chunk[:,:,1::2,::2]).sum(axis=-1) * h
    
    # odd col local correction
    wh = interp1d(wH, order=order) 
    wh[:,:,::2] += (Khh_correction[:,:,::2] * uh_chunk[:,:,::2]).sum(axis=-1) * h

    return wh

def calculate_kernel_band_diff(KHH, Khh, band_idx, order=2):
    idx_i, idx_j = band_idx
    Khh_pad = F.pad(Khh, (1,1,1,1))

    # interp KHH to Khh_interp
    Khh_even_interp = interp1d_cols(KHH, order=order)
    Khh_interp = interp1d_rows(Khh[:,:,1::2], order=order)    
    Khh_interp[:,:,1::2] = Khh_even_interp

    # Khh_interp = interp1d_rows(Khh_even_interp, order=order)

    Khh_interp_pad = F.pad(Khh_interp, (1,1,1,1))

    # diff on band
    Khh_error = (Khh_pad - Khh_interp_pad)[:,:,idx_i, idx_j]

    return Khh_error


def SmoothKernelMLMM(uh, Khh, h, k=3):
    # calculate H
    H = h * 2**k

    if k == 0:
        wh = h * torch.einsum('bcmn, bcn->bcm', Khh, uh)
    else:
        # restrict uh
        for i in range(k):
            if i == 0:
                uH = restrict1d(uh)
            else:
                uH = restrict1d(uH)
        
        # injection Khh
        KHH = Khh[:,:,2**k-1::2**k, 2**k-1::2**k]

        # integral transform on coarse grid H*KHH*uH
        wH = H * torch.einsum('bcmn, bcn->bcm', KHH, uH)

        # interp wH 
        for i in range(k):
            if i == 0:
                wh = interp1d(wH)
            else:
                wh = interp1d(wh)
    
    return wh

def SingularSmoothKernelMLMM(KHH, uh, h, Khh_correction_lst, nbr_idx_lst, order=2):
    k = len(Khh_correction_lst)
    uh_chunk_lst = []

    for i in range(k):
        # fetch nbrs and correct index which is out of domain        
        uh_pad = F.pad(uh, (1,1))
        uh_chunk = uh_pad[:,:,nbr_idx_lst[i]]
        uh_chunk_lst.append(uh_chunk)
        uh = restrict1d(uh, order=order)
        h = h*2

    # coarse evaluation
    H = h
    uH = uh 
    wH = multi_summation(KHH, uH, H)
    k = len(Khh_correction_lst)
    uh_chunk_lst = uh_chunk_lst[::-1]
    
    # multi-level correction
    for i in range(k):
        H = H/2
        wh = local_correction(wH, Khh_correction_lst[i], uh_chunk_lst[i], H, order=order)
        wH = wh

    return wh

def SingularSmoothKernelMLMM_full(uh, Khh, h, k=3, order=2, m=7):
    Khh_correction_lst = []
    uh_chunk_lst = []
    nbr_idx_lst = []
    nf = Khh.shape[-1]

    for i in range(k):
        n = Khh.shape[-1]
        if n <= nf ** 0.5:
            print(" N is {:}, less than {:}^0.5".format(n, nf))

        # fetch nbrs and correct index which is out of domain        
        idx_i, idx_j = fetch_nbrs(n, m=m)
        idx_j[idx_j < 0] = 0
        idx_j[idx_j > n] = 0
        nbr_idx_lst.append(idx_j)

        uh_pad = F.pad(uh, (1,1))
        uh_chunk = uh_pad[:,:,idx_j]
        uh_chunk_lst.append(uh_chunk)

        # coarsen Khh by injection and uh by restriction
        KHH = Khh[:,:,1::2,1::2] 
        uH = restrict1d(uh, order=order)
        H = h*2

        # local correction on band area
        Khh_correction = calculate_kernel_band_diff(KHH, Khh, (idx_i, idx_j), order=order)
        Khh_correction_lst.append(Khh_correction)

        # update
        Khh = KHH
        uh = uH 
        h = H
            
    # coarsest approximation
    wh = multi_summation(Khh, uh, h)
    
    # reverse Khh_correction_list and uh_chunk_lst
    Khh_correction_lst = Khh_correction_lst[::-1]
    uh_chunk_lst = uh_chunk_lst[::-1]

    for i in range(k):
        h = h/2
        wh = local_correction(wh, Khh_correction_lst[i], uh_chunk_lst[i], h, order=order)

    
    return wh, Khh, Khh_correction_lst, nbr_idx_lst


if __name__ == '__main__':
    l = 13
    n = 2**l - 1
    xh = torch.linspace(0, torch.pi, n+2)[1:-1]
    h = xh[0]
    nk = 10
    niter = 50

    # ----------------------------------------
    # smooth kernel test 
    # ----------------------------------------

    # input function
    uh = torch.sin(xh)**2

    # kernel function
    gh_X, gh_Y = torch.meshgrid(xh, xh)
    Khh = torch.cos(gh_Y - gh_X)

    # # output function(numeric)
    wh_gt = h * Khh @ uh 

    # # analytics
    # wh_gt = torch.sin(xh) * 4/3

    for k in range(nk):
        start_time = time.time()
        
        for i in range(niter):
            wh = SmoothKernelMLMM(uh[None][None], Khh[None][None], h, k)
        
        end_time = time.time()
        elapsed = (end_time - start_time)/niter

        L1 = (wh[0,0] - wh_gt).abs().sum()/(n+1)
        print('nh : {:>6} - nH : {:>6} - L1 : {:>.4e} - time : {:>.4f}s '.format(n, 2**(l-k)-1, L1.item(), elapsed))

        if n**0.5 > 2**(l-k)-1:
            print('---------------------------')

    # ------------------------------------------