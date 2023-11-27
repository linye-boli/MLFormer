import pdb 
import torch 
import time
from utils import restrict1d, interp1d

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

def SingularSmoothKernelMLMM(uh, Khh, h, k=3):
    pass 

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