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

def SingularSmoothKernelMLMM(KHH, uh, Khh_banddiff_lst, idx_j_lst, h, k, order=2):
    assert len(idx_j_lst) == k
    uh_band_lst = []

    for i in range(k):
        uh_band = uh[:,:,idx_j_lst[i]]
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
        wh_even[:,:,1:-1] += wh_even_corr[:,:,1:-1]
        wh = interp1d(wh_even, order=order)
        wh[:,:,1::2] = wh[:,:,1::2] + (Khh_banddiff_lst[i][1] * uh_band_lst[i][1]).sum(axis=-1)*h

        wh_ml.append(wh)

    return wh_ml

def SingularSmoothKernelMLMM_full(uh, Khh, h, k=3, order=2):
    # Khh_hom = Khh.clone()
    # Khh_hom[:,:,1:-1,1:-1] = 0
    # wh_hom = multi_summation(Khh_hom, uh, h)
    # wh_hom = torch.zeros_like(uh)

    Khh[:,:,0] = 0
    Khh[:,:,-1] = 0
    Khh[:,:,:,0] = 0
    Khh[:,:,:,-1] = 0

    corr_lst = []
    uh_lst = []
    for i in range(k):
        
        KHh = injection1d_rows(Khh) 
        KHH = injection1d_cols(KHh)
        KHh_smooth = interp1d_cols(KHH) # \tilde{K}
        KHh_corr_even = KHh - KHh_smooth

        Khh_smooth = interp1d_rows(KHh)[:,:,1::2]
        Khh_corr_odd = Khh[:,:,1::2] - Khh_smooth

        corr_lst.append([KHh_corr_even, Khh_corr_odd])
        uh_lst.append(uh)
        
        uh = restrict1d(uh)
        h = h*2
        Khh = KHH
    
    corr_lst = corr_lst[::-1]
    uh_lst = uh_lst[::-1]
    wh = multi_summation(Khh, uh, h) 
    wh_ml = [wh]
    for i in range(k):
        h = h / 2
        wh_even_corr = multi_summation(corr_lst[i][0], uh_lst[i], h)
        wh_even = wh + wh_even_corr
        wh_ = interp1d(wh_even)
        wh_[:,:,1::2] = wh_[:,:,1::2] + multi_summation(corr_lst[i][1], uh_lst[i], h)
        wh = wh_
        wh_ml.append(wh)

    return wh_ml, Khh
    
def SingularSmoothKernelMLMM_local(uh, Khh, h, k=3, order=2, m=7):
    Khh_banddiff_lst = []
    uh_band_lst = []
    idx_j_lst = []

    for i in range(k):
        # evaluate kernel function on coarse grid
        KHh = injection1d_rows(Khh) 
        KHH = injection1d_cols(KHh)
        KHh = injection1d_rows(Khh) 

        # smooth approximation of kernel function
        KHh_smooth = interp1d_cols(injection1d_cols(Khh), order=order) # \tilde{K}
        Khh_smooth = interp1d_rows(KHh, order=order)

        # fetch nbr idx
        n = Khh.shape[-1] 
        idx_i, idx_j = fetch_nbrs(n, m) 
        idx_j[idx_j < 0] = 0 
        idx_j[idx_j > n-1] = n-1 
        idx_j_lst.append(idx_j)

        # band diff between fine kernel and smooth approximation
        KHh_banddiff_even = (Khh - KHh_smooth)[:,:,idx_i, idx_j][:,:,::2]
        Khh_banddiff_odd = (Khh - Khh_smooth)[:,:,idx_i, idx_j][:,:,1::2]

        # correction kernels
        Khh_banddiff_lst.append(
            [KHh_banddiff_even[:,:,:,::2], Khh_banddiff_odd])

        # uh band
        uh_band = uh[:,:,idx_j]
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

    # coarsest approximation
    wh = multi_summation(Khh, uh, h)    
    wh_ml = [wh]

    # multi-level correction
    for i in range(k):
        h = h/2        
        wh_even_corr = (Khh_banddiff_lst[i][0]*uh_band_lst[i][0]).sum(axis=-1)*h
        wh_even = wh 
        wh_even[:,:,1:-1] += wh_even_corr[:,:,1:-1]
        wh = interp1d(wh_even, order=order)
        wh[:,:,1::2] = wh[:,:,1::2] + (Khh_banddiff_lst[i][1] * uh_band_lst[i][1]).sum(axis=-1)*h

        wh_ml.append(wh)
    
    return wh_ml, Khh, Khh_banddiff_lst, idx_j_lst

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
        Khh[:,:,idx_i_lst[i][1::2], idx_j_lst[i][1::2]] += Khh_banddiff_lst[i][1]
        Khh_lst.append(Khh)

    return Khh_lst


# class MLP(nn.Module):
#     def __init__(self, in_channels, out_channels, mid_channels):
#         super(MLP, self).__init__()
#         self.mlp1 = nn.Linear(in_channels, mid_channels)
#         self.mlp2 = nn.Linear(mid_channels, mid_channels)
#         self.mlp3 = nn.Linear(mid_channels, out_channels)

#     def forward(self, x):
#         x = self.mlp1(x)
#         x = F.relu(x)
#         x = self.mlp2(x)
#         x = F.relu(x)
#         x = self.mlp3(x)
        
#         return x

# class SpectralModel(nn.Module):
#     def __init__(self, rank, modes1, modes2, m, n):
#         super(SpectralModel, self).__init__()
#         self.m = m 
#         self.n = n
#         self.modes1 = modes1
#         self.modes2 = modes2 
#         self.rank = rank

#         self.spectral_kernel1 = nn.Parameter(1/rank * torch.rand(rank, modes1, modes2, dtype=torch.cfloat))
#         self.spectral_kernel2 = nn.Parameter(1/rank * torch.rand(rank, modes1, modes2, dtype=torch.cfloat))
#         self.mlp = MLP(rank, 1, 128)

        
#     def forward(self, x):
#         if self.n%2 == 0:
#             n = self.n//2 + 1
#         else:
#             n = self.n//2 + 2

#         ftK = torch.zeros(self.rank, self.m, n, dtype=torch.cfloat, device=x.device)
#         ftK[:, :self.modes1, :self.modes2] = self.spectral_kernel1
#         ftK[:, -self.modes1:, :self.modes2] = self.spectral_kernel2
#         K = torch.fft.irfft2(ftK, s=(self.m, self.n))
#         # if self.n % 2 == 1:
#         #     K = K[:,:-1]
        
#         K = rearrange(K, 'r m n -> (n m) 1 r')
#         K = self.mlp(K)
#         K = rearrange(K, '(m n) 1 1 -> m n', m=self.m, n=self.n)

#         return K 

# class MLPLowRankModel(nn.Module):
#     def __init__(self, rank, m, n):
#         super(MLPLowRankModel, self).__init__()
#         self.m = m
#         self.n = n
#         self.phi = MLP(1, rank, 128)
#         self.psi = MLP(1, rank, 128)
#         # self.mlp = MLP(rank, 1, 128)
#     def forward(self, u):
    
#         x_phi = torch.linspace(0,1,self.m).to(u)
#         x_psi = torch.linspace(0,1,self.n).to(u)
#         x_phi = rearrange(x_phi, 'm -> m 1')
#         x_psi = rearrange(x_psi, 'm -> m 1')
        
#         phi = self.phi(x_phi) # n rank
#         psi = self.psi(x_psi) # n rank
#         K = torch.einsum('nr,mr->nm', phi, psi)

#         # K = self.mlp(K)
#         # K = rearrange(K, 'b n m 1-> m n')

#         return K 

# class LowRankModel(nn.Module):
#     def __init__(self, rank, m, n):
#         super(LowRankModel, self).__init__()
#         self.m = m
#         self.n = n
#         self.phi = nn.Parameter(torch.zeros(m, rank))
#         self.psi = nn.Parameter(torch.zeros(n, rank))
#         self.mlp = MLP(rank, 1, 128)
    
#     def forward(self, x):
#         K = torch.einsum("mr, nr->mnr", self.phi, self.psi)
#         K = rearrange(K, 'm n r -> (n m) 1 r')
#         K = self.mlp(K)
#         K = rearrange(K, '(m n) 1 1 -> m n', m=self.m, n=self.n)

#         return K 




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
    