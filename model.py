import torch 
import torch.nn as nn 
from mlmm import fetch_nbrs, SingularSmoothKernelMLMM

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

class MLFormer(nn.Module):
    def __init__(self, rank, m, k, l, h, order=2, mtype='lowrank', correction=True):
        super(MLFormer, self).__init__()

        self.k = k 
        self.m = m 
        self.l = l 
        self.h = h
        self.order = order
        self.correction = correction
        self.mtype = mtype

        self.nf = 2**l + 1
        N = 2**(l-k) + 1 # number of points on coarsest level

        parameter_dict = {}
        if mtype == 'lowrank':
            parameter_dict['KHH'] = LowRankModel(rank, N, N)
        elif mtype == 'mlp_lowrank':
            # parameter_dict['KHH'] = LowRankModel(rank, N, N)
            parameter_dict['KHH'] = MLPLowRankModel(rank, N, N)
        elif mtype == 'fourier':
            parameter_dict['KHH'] = SpectralModel(rank, 12, 12, N, N)
        

        if correction:        
            for i in range(k):
                if i != 0:
                    N = 2*N - 1      

                if mtype == 'lowrank':
                    Khhc_even = LowRankModel(rank, N, m+1)
                    Khhc_odd = LowRankModel(rank, N-1, 2*m+1)
                elif mtype == 'mlp_lowrank':
                    Khhc_even = MLPLowRankModel(rank, N, m+1)
                    Khhc_odd = MLPLowRankModel(rank, N-1, 2*m+1)
                elif mtype == 'fourier':
                    Khhc_even = SpectralModel(rank, 12,3, N, m+1)
                    Khhc_odd = SpectralModel(rank, 12,3, N-1, 2*m+1)

                parameter_dict[f'Khhc_{i}_even'] = Khhc_even
                parameter_dict[f'Khhc_{i}_odd'] = Khhc_odd

            self.nbr_idx_lst = self.fetch_ml_nbrs()

            if mtype == 'lowrank':
                parameter_dict['Khh_bd'] = LowRankModel(rank, self.nf, 2)
            if mtype == 'mlp_lowrank':
                parameter_dict['Khh_bd'] = MLPLowRankModel(rank, self.nf, 2)
            elif mtype == 'fourier':
                parameter_dict['Khh_bd'] = SpectralModel(rank, 12,1,self.nf,2)
        
        self.models = nn.ModuleDict(parameter_dict)
    
    def fetch_ml_nbrs(self):
        nbr_idx_lst = []
        for i in range(self.k):
            if i == 0:
                n = self.nf - 1
            else:
                n = n// 2

            if n <= self.nf ** 0.5:
                print(" N is {:}, less than {:}^0.5".format(n, self.nf))

            # fetch nbrs and correct index which is out of domain        
            _, idx_j = fetch_nbrs(n, self.m)
            idx_j[idx_j < 0] = 0
            idx_j[idx_j > n] = 0
            nbr_idx_lst.append(idx_j)

        return nbr_idx_lst

    def forward(self, uh):
        KHH = self.models['KHH'](uh)
        Khh_correction_lst = []
        for i in range(self.k):
            Khhc_even = self.models[f'Khhc_{i}_even'](uh)
            Khhc_odd = self.models[f'Khhc_{i}_odd'](uh)       
            Khh_correction_lst.append((Khhc_even, Khhc_odd))

        Khh_bd = self.models['Khh_bd'](uh)
        w_bd = torch.einsum('nm,bcn->bcm', Khh_bd, uh)*self.h
        w_bd = [w_bd[:,:,[0]], w_bd[:,:,[1]]]
        
        wh_ml = SingularSmoothKernelMLMM(KHH[None][None], uh, self.h, Khh_correction_lst, self.nbr_idx_lst, w_bd, order=self.order)
        return wh_ml, KHH, Khh_correction_lst