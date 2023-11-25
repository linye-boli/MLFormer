import torch 
import torch.nn.functional as F
from einops import rearrange

def interp1d(vH, order=2):
    # vH : (batch, 1, len)

    if order == 2:
        kernel = torch.tensor([[[1., 2., 1.]]]).to(vH)
        w = 1/2 
        s = 2
    
    vh = w * F.conv_transpose1d(vH, kernel, stride=s)
    return  vh 

def restrict1d(vh, order=2):
    # vh : (batch, 1, len)
    if order == 2:
        kernel = torch.tensor([[[1., 2., 1.]]]).to(vh)
        w = 1/4
        s = 2

    vH = w * F.conv1d(vh, kernel, stride=s)
    return vH

if __name__ == '__main__':

    # test interp1d
    vH = torch.tensor([[0.1, 0.5], [0.8, -0.2], [-0.4, 0.6]])
    IHh = torch.tensor([
        [1., 0., 0.],
        [2., 0., 0.],
        [1., 1., 0.],
        [0., 2., 0.],
        [0., 1., 1.],
        [0., 0., 2.],
        [0., 0., 1.],
    ])   

    vh_ = 0.5 * IHh @ vH 
    print('mat result : ')
    print(vh_)

    vH = rearrange(vH, 'n b -> b 1 n')
    vh_ = interp1d(vH, order=2)
    print('deconv result : ')
    print(vh_[:,0].T)

    vh = torch.tensor([
        [0.1, 0.5], [0.8, -0.2], [-0.4, 0.6], [0.1, 0.5], [0.8, -0.2],[-0.4, 0.6], [0.1, 0.5]])
    vH_ = 0.25 * IHh.T @ vh
    print('mat result : ')
    print(vH_)

    vh = rearrange(vh, 'n b -> b 1 n')
    vH_ = restrict1d(vh, order=2)
    print('conv result : ')
    print(vH_[:,0].T)