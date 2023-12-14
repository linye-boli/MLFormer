import torch 
import torch.nn as nn 
from einops import rearrange
from utils import (
    injection1d_rows, injection1d_cols, 
    restrict1d, interp1d,
    multi_summation)
from mlmm import fetch_nbrs

class Rational(torch.nn.Module):
    """Rational Activation function.
    It follows:
    `f(x) = P(x) / Q(x),
    where the coefficients of P and Q are initialized to the best rational 
    approximation of degree (3,2) to the ReLU function
    # Reference
        - [Rational neural networks](https://arxiv.org/abs/2004.01902)
    """
    def __init__(self):
        super().__init__()
        self.coeffs = torch.nn.Parameter(torch.Tensor(4, 2))
        self.reset_parameters()

    def reset_parameters(self):
        self.coeffs.data = torch.Tensor([[1.1915, 0.0],
                                    [1.5957, 2.383],
                                    [0.5, 0.0],
                                    [0.0218, 1.0]])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.coeffs.data[0,1].zero_()
        exp = torch.tensor([3., 2., 1., 0.], device=input.device, dtype=input.dtype)
        X = torch.pow(input.unsqueeze(-1), exp)
        PQ = X @ self.coeffs
        output = torch.div(PQ[..., 0], PQ[..., 1])
        return output

# A simple feedforward neural network
class MLP(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(MLP, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j+1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x

class NMLK(torch.nn.Module):
    def __init__(self, layers, l=13, k=3, m=7, order=2):
        super(NMLK, self).__init__()
        self.k = k
        self.m = m
        self.l = l
        self.n = 2**l + 1
        self.order=order
        model_dict = {}
        model_dict['KHH'] = MLP(layers, Rational)
        for i in range(k):
            model_dict[f'Khh_band_even_{i+1}'] = MLP(layers, Rational)
            model_dict[f'Khh_band_odd_{i+1}'] = MLP(layers, Rational)

        self.models = nn.ModuleDict(model_dict)
        self.idx_j_lst = self.get_nbrs()

    def ml_grids(self, x, x_nbrs):
        assert len(x) == self.n

        # boundary coords on finest level
        bd_coords = torch.cartesian_prod(x[[0,-1]], x)

        # local coords on each level        
        local_coords = torch.cartesian_prod(x, x_nbrs)
        local_coords = rearrange(local_coords, '(n m) c -> 1 c n m', n = self.n, m=2*self.m+1, c=2)
        local_coords_lst = []
        for i in range(self.k):
            local_coords_even = rearrange(local_coords[0,:,::2,::2], 'c m n -> m n c')
            local_coords_odd = rearrange(local_coords[0,:,1::2], 'c m n -> m n c')
            local_coords_lst.append([local_coords_even, local_coords_odd])
            local_coords = injection1d_rows(local_coords)
        local_coords_lst = local_coords_lst[::-1]

        # full coords on coarest level
        xH = local_coords[0,0,:,0]
        gHx, gHy = torch.meshgrid(xH, xH, indexing='ij')
        gHH = rearrange([gHx, gHy], 'b m n -> m n b')

        return gHH, bd_coords, local_coords_lst

    def ml_kernel(self, gHH, bd_coords, local_coords_lst):
        # approximate KHH by MLP
        KHH = self.models['KHH'](gHH)
        KHH = rearrange(KHH, 'm n c -> 1 c m n')

        # approximate Khh_bd by MLP
        Khh_bd = self.models['KHH'](bd_coords)
        Khh_bd = [Khh_bd[:self.n], Khh_bd[self.n:]]
        Khh_bd = [rearrange(K, 'm n -> 1 1 n m') for K in Khh_bd]

        # approximate Khh_banddiff_lst by MLP
        Khh_banddiff_lst = []
        for i in range(self.k):
            Khh_banddiff_even = self.models[f'Khh_band_even_{i+1}'](local_coords_lst[i][0])
            Khh_banddiff_odd = self.models[f'Khh_band_odd_{i+1}'](local_coords_lst[i][1])

            Khh_banddiff_even = rearrange(Khh_banddiff_even, 'm n c -> 1 c m n')
            Khh_banddiff_odd = rearrange(Khh_banddiff_odd, 'm n c -> 1 c m n')

            Khh_banddiff_lst.append([Khh_banddiff_even, Khh_banddiff_odd])
        
        return KHH, Khh_bd, Khh_banddiff_lst
    
    def get_nbrs(self):
        idx_j_lst = []
        for i in range(self.k):
            n = 2**(self.l - i)+1
            _, idx_j = fetch_nbrs(n, self.m)
            idx_j[idx_j < 0] = 0 
            idx_j[idx_j > n-1] = n-1 
            idx_j_lst.append(idx_j)        
        return idx_j_lst
    
    def prepare_u_and_wb(self, uh, Khh_bd, h):
        assert len(self.idx_j_lst) == self.k
        uh_band_lst = []
        lb_lst = []
        rb_lst = []
        for i in range(self.k):
            n = uh.shape[-1]

            # prepare uh_band
            idx_mask = (self.idx_j_lst[i] >= 0) & (self.idx_j_lst[i] <= n-1)
            uh_band = uh[:,:,self.idx_j_lst[i]] * idx_mask
            uh_band_even = uh_band[:,:,::2]
            uh_band_odd = uh_band[:,:,1::2]
            uh_band_lst.append([uh_band_even[:,:,:,::2], uh_band_odd])
            
            # calculate w boundaries
            w_lb = multi_summation(Khh_bd[0], uh, h)
            w_rb = multi_summation(Khh_bd[1], uh, h)
            lb_lst.append(w_lb)
            rb_lst.append(w_rb)

            # coarsen Khh boundaries
            Khh_bd[0] = injection1d_cols(Khh_bd[0])
            Khh_bd[1] = injection1d_cols(Khh_bd[1])

            # coarsen uh
            uh = restrict1d(uh, order=self.order)
            # coarsen h      
            h = h*2

        boundary_lst = [lb_lst[::-1], rb_lst[::-1]]
        uh_band_lst = uh_band_lst[::-1]

        return boundary_lst, uh_band_lst, uh, h

    def forward(self, x, x_nbrs):
        # normalize grid to [0, 1]
        x = (x - x[0])/(x[-1] - x[0])

        # coarsest level full grids : gHH
        # finest level boundary grids : bd_coords
        # local coords on each level grids : local_coords_lst
        gHH, bd_coords, local_coords_lst = self.ml_grids(x, x_nbrs)

        # Prepare KHH, Khh_bd, Khh_banddiff_lst
        Khh, Khh_bd, Khh_banddiff_lst = self.ml_kernel(gHH, bd_coords, local_coords_lst) # here Khh is KHH
        # Prepare u_band and w_bd
        # boundary_lst, uh_band_lst, uh, h = self.prepare_u_and_wb(uh, Khh_bd, h) # here uh is uH

        # # coarsest approximation
        # wh = multi_summation(Khh, uh, h)
        # wh_ml = [wh]

        # # multi-level correction
        # for i in range(self.k):
        #     h = h/2        
        #     wh_even_corr = (Khh_banddiff_lst[i][0]*uh_band_lst[i][0]).sum(axis=-1)*h
        #     wh += wh_even_corr
        #     wh[:,:,[0]] = boundary_lst[0][i]
        #     wh[:,:,[-1]] = boundary_lst[1][i]
        #     wh = interp1d(wh, order=self.order)
        #     wh[:,:,1::2] = wh[:,:,1::2] + (Khh_banddiff_lst[i][1] * uh_band_lst[i][1]).sum(axis=-1)*h
        #     wh_ml.append(wh)

        return Khh, Khh_banddiff_lst