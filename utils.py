import numpy as np
import torch 
import torch.nn.functional as F
from einops import rearrange, repeat

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
    
    vh = w * F.conv_transpose1d(vH, kernel, stride=s, padding=p)
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
    

    vH = w * F.conv1d(vh, kernel, stride=s, padding=p)
    return vH

def interp1d_cols(KhH, order=2):
    # KhH : (batch, c, i, J)

    bsz, c, i, J = KhH.shape
    KhH = rearrange(KhH, 'b c i J -> (b i) c J')
    Khh = interp1d(KhH, order=order)
    Khh = rearrange(Khh, '(b i) c j-> b c i j', b = bsz, c=c, i=i, j=2*J+1)

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

def L1Norm(est, ref):
    b, c, n = est.shape
    return (est - ref).abs().sum().item()/(n+1)/b

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


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
    print('ord2 : ', L1Norm(vh_ord2,vh))
    print('ord4 : ', L1Norm(vh_ord4,vh))
    print('ord6 : ', L1Norm(vh_ord6,vh))
    
    print('matmul interp error : ')
    print('ord2 : ', L1Norm(vh_ord2_mat,vh))
    print('ord4 : ', L1Norm(vh_ord4_mat,vh))
    print('ord6 : ', L1Norm(vh_ord6_mat,vh))
    
    print('conv restrict error : ')
    print('ord2 : ', L1Norm(vH_ord2,vH))
    print('ord4 : ', L1Norm(vH_ord4,vH))
    print('ord6 : ', L1Norm(vH_ord6,vH))
    
    print('matmul restrict error : ')
    print('ord2 : ', L1Norm(vH_ord2_mat,vH))
    print('ord4 : ', L1Norm(vH_ord4_mat,vH))
    print('ord6 : ', L1Norm(vH_ord6_mat,vH))
    