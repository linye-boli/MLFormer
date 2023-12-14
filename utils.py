import os 
import numpy as np
import torch 
import torch.nn.functional as F
from einops import rearrange, repeat
import pandas as pd

def mlgrid1d(xh, k):
    xh_ml = [xh]
    for _ in range(k):
        xh = injection1d(xh)
        xh_ml.append(xh)
        
    return xh_ml[::-1]

def mlgrid2d(ghh, k):
    ghh_ml = [ghh]
    for _ in range(k):
        ghH = injection1d_cols(ghh)
        ghh = injection1d_rows(ghH)
        ghh_ml.append(ghh)
    
    return ghh_ml[::-1]

def injection2d(Khh):
    KhH = torch.cat([Khh[...,[0]], Khh[...,1:-1][...,1::2], Khh[...,[-1]]], axis=-1)
    KHH = torch.cat([KhH[...,[0],:], KhH[...,1:-1,:][...,1::2,:], KhH[...,[-1],:]], axis=-2)
    return KHH

def injection1d_cols(Khh):
    KhH = torch.cat([Khh[...,[0]], Khh[...,1:-1][...,1::2], Khh[...,[-1]]], axis=-1)
    return KhH

def injection1d_rows(Khh):
    # KhH : (batch, c, H, h)

    Khh = rearrange(Khh, 'b c I j -> b c j I')
    KHh = injection1d_cols(Khh)
    KHh = rearrange(KHh, 'b c j i -> b c i j')

    return KHh

def injection1d(vh):
    vH = torch.cat([vh[...,[0]], vh[...,1:-1][...,1::2], vh[...,[-1]]], axis=-1)
    return vH

def interp1d_mat(n, order=2):
    if order == 2:
        mat = torch.zeros((2*n+1, n))
        kernel = torch.tensor([1., 2., 1.])/2
        klen = 3
    elif order == 4:
        mat = torch.zeros((2*n+1+4, n))
        kernel = torch.tensor([-1., 0, 9, 16, 9, 0, -1.])/16
        klen = 7
    elif order == 6:
        mat = torch.zeros((2*n+1+8, n))
        kernel = torch.tensor([3., 0, -25, 0, 150, 256, 150, 0, -25, 0, 3])/256
        klen = 11

    for i in range(n):
        mat[2*i:2*i+klen, i] = kernel 
        
    if order == 4:
        mat = mat[2:-2]
    elif order == 6:
        mat = mat[4:-4]

    return mat

def restrict1d_mat(n, order=2):
    interpmat = interp1d_mat(n, order)/2
    return interpmat.T

def interp1d_matmul(vH, interpmat=None, order=2):
    n = vH.shape[-1]
    if interpmat is None:
        interpmat = interp1d_mat(n, order)
    return torch.einsum('mn, bcn->bcm', interpmat, vH)

def restrict1d_matmul(vh, restrictmat=None, order=2):
    n = vh.shape[-1]
    if restrictmat is None:
        restrictmat = restrict1d_mat((n-1)//2, order)
    return torch.einsum('mn, bcn->bcm', restrictmat, vh)

def interp1d(vH, order=2):
    # vH : (batch, c, H)

    if order == 2:
        kernel = torch.tensor([[[1., 2., 1.]]]).to(vH)
        w = 1/2 
        s = 2
        p = 0

    if order == 4:
        kernel = torch.tensor([[[-1,0,9,16,9,0,-1]]]).to(vH)
        w = 1/16
        s = 2
        p = 2
    
    if order == 6:
        kernel = torch.tensor([[[3, 0, -25, 0, 150, 256, 150, 0, -25, 0, 3]]]).to(vH)
        w = 1/256
        s = 2 
        p = 4
    
    vh = w * F.conv_transpose1d(vH, kernel, stride=s, padding=p)[..., 1:-1]
    return  vh 

def restrict1d(vh, order=2):
    # vh : (batch, c, h)

    if order == 2:
        kernel = torch.tensor([[[1., 2., 1.]]]).to(vh)
        w = 1/4
        s = 2
        p = 0
    
    if order == 4:
        kernel = torch.tensor([[[-1,0,9,16,9,0,-1]]]).to(vh)
        w = 1/32
        s = 2
        p = 2
    
    if order == 6:
        kernel = torch.tensor([[[3, 0, -25, 0, 150, 256, 150, 0, -25, 0, 3]]]).to(vh)
        w = 1/512
        s = 2
        p = 4

    vH = w * F.conv1d(vh[...,1:-1], kernel, stride=s, padding=p)
    vH = torch.cat([vh[...,[0]], vH, vh[...,[-1]]], axis=-1)
    return vH

def interp1d_cols(KhH, order=2):
    # KhH : (batch, c, i, J)

    bsz, c, i, J = KhH.shape
    KhH = rearrange(KhH, 'b c i J -> (b i) c J')
    Khh = interp1d(KhH, order=order)
    Khh = rearrange(Khh, '(b i) c j-> b c i j', b = bsz, c=c, i=i, j=2*J-1)

    return Khh

def interp1d_rows(KHh, order=2):
    # KhH : (batch, c, H, h)

    KhH = rearrange(KHh, 'b c I j -> b c j I')
    Khh = interp1d_cols(KhH, order)
    Khh = rearrange(Khh, 'b c j i -> b c i j')

    return Khh

def multi_summation(K, u, h):
    # KHH : (batch, c, m, n)
    # u : (batch, c, n)
    # h : float scalar

    return h * torch.einsum('bcmn, bcn-> bcm', K, u)
    
def numeric_integ(K, u, h):
    # KHH : (batch, c, m, n)
    # u : (batch, c, n)
    # h : float scalar
    
    return h * torch.einsum('bcmn, bcn-> bcm', K[:,:,:,1:-1], u[:,:,1:-1]) + h/2 * torch.einsum('bcmn, bcn-> bcm', K[:,:,:,[0]], u[:,:,[0]]) + h/2 * torch.einsum('bcmn, bcn-> bcm', K[:,:,:,[-1]], u[:,:,[-1]])

def l1_norm(est, ref):
    if len(est.shape) == 2:
        b, n = est.shape 
    elif len(est.shape) == 3:
        b, c, n = est.shape 
    return ((est - ref).abs().sum(axis=-1)/(n+2)).mean() # here n+2 indicates i=0,1,2,...2^13
    
def rl2_error(est, ref):
    if len(est.shape) == 2:
        b, n = est.shape 
    elif len(est.shape) == 3:
        b, c, n = est.shape 
    return ((((est - ref)**2).sum(axis=-1))**0.5 / ((ref**2).sum(axis=-1))**0.5).mean()

def matrl2_error(est, ref):
    est = est.reshape(1,-1)
    ref = ref.reshape(1,-1)
    return rl2_error(est, ref)

def ml_rl2_error(est, ref, k, order=2):
    est = est[::-1]
    for i in range(k+1):
        if i == 0:
            rl2 = rl2_error(est[i], ref)
        else:
            ref = restrict1d(ref, order=order)
            rl2 += rl2_error(est[i], ref)
    return rl2

def init_records(task_nm, log_root, model_nm):
    exp_root = os.path.join(log_root, task_nm, model_nm)
    os.makedirs(exp_root, exist_ok=True)

    hist_outpath = os.path.join(exp_root, 'hist.csv')
    pred_outpath = os.path.join(exp_root, 'pred.csv')
    model_operator_outpath = os.path.join(exp_root, 'model_best_operator.pth')
    model_kernel_outpath = os.path.join(exp_root, 'model_best_kernel.pth')
    
    return hist_outpath, pred_outpath, model_operator_outpath, model_kernel_outpath

def save_hist(hist_outpath, train_hist, test_hist, kernel_hist=None):
    if kernel_hist is None:
        log_df = pd.DataFrame({'train_rl2': train_hist, 'test_rl2': test_hist})
    else:
        log_df = pd.DataFrame({'train_rl2': train_hist, 'test_rl2': test_hist, 'test_matrl2': kernel_hist})
    log_df.to_csv(hist_outpath, index=False)
    print('save train-test log at : ', hist_outpath)

def save_preds(pred_outpath, preds):
    preds = np.array(preds)
    preds = rearrange(preds, 'n b l -> (n b) l')
    np.savetxt(pred_outpath, preds, delimiter=',')
    print('save test predictions at : ', pred_outpath)

if __name__ == '__main__':

    # test interp1d
    l = 8
    n = 2**l - 1
    lb = 0
    ub = 2*np.pi
    xh = torch.linspace(lb, ub, n+2)[1:-1][None][None]
    xH = xh[:,:,1::2]
    vh = torch.sin(xh)
    vH = torch.sin(xH)

    vh_ord2 = interp1d(vH, order=2)
    vh_ord4 = interp1d(vH, order=4)
    vh_ord6 = interp1d(vH, order=6)

    vh_ord2_mat = interp1d_matmul(vH, order=2)
    vh_ord4_mat = interp1d_matmul(vH, order=4)
    vh_ord6_mat = interp1d_matmul(vH, order=6)

    vH_ord2 = restrict1d(vh, order=2)
    vH_ord4 = restrict1d(vh, order=4)
    vH_ord6 = restrict1d(vh, order=6)

    vH_ord2_mat = restrict1d_matmul(vh, order=2)
    vH_ord4_mat = restrict1d_matmul(vh, order=4)
    vH_ord6_mat = restrict1d_matmul(vh, order=6)

    print('deconv interp error(L1Norm) : ')
    print('ord2 : ', l1_norm(vh_ord2,vh))
    print('ord4 : ', l1_norm(vh_ord4,vh))
    print('ord6 : ', l1_norm(vh_ord6,vh))
    
    print('matmul interp error : ')
    print('ord2 : ', l1_norm(vh_ord2_mat,vh))
    print('ord4 : ', l1_norm(vh_ord4_mat,vh))
    print('ord6 : ', l1_norm(vh_ord6_mat,vh))
    
    print('conv restrict error : ')
    print('ord2 : ', l1_norm(vH_ord2,vH))
    print('ord4 : ', l1_norm(vH_ord4,vH))
    print('ord6 : ', l1_norm(vH_ord6,vH))
    
    print('matmul restrict error : ')
    print('ord2 : ', l1_norm(vH_ord2_mat,vH))
    print('ord4 : ', l1_norm(vH_ord4_mat,vH))
    print('ord6 : ', l1_norm(vH_ord6_mat,vH))
    