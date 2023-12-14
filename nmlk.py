import torch 
from utils import (
    multi_summation,
    rl2_error, ml_rl2_error, matrl2_error,
    init_records, save_hist, save_preds)
from dataset import load_dataset_1d, reference_test
from tqdm import trange
import argparse
from model import NMLK
from mlmm import SingularSmoothKernelMLMM_local, SingularSmoothKernelReconstruction
from datetime import datetime
import json

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Train a mlformer operator learning in 1d")
    parser.add_argument('--device', type=int, default=0,
                        help='device id.')
    parser.add_argument('--task', type=str, default='cosine',
                        help='dataset name. (burgers, poisson, cosine, lnabs)')
    parser.add_argument('--epochs_adam', type=int, default=400,
                        help='epochs for train with adam.')
    parser.add_argument('--lr_adam', type=float, default=1e-3,
                        help='learning rate of adam optimizer')
    parser.add_argument('--epochs_lbfgs', type=int, default=20,
                        help='epochs for train with lbfgs.')
    parser.add_argument('--lr_lbfgs', type=float, default=1e-2,
                        help='learning rate of lbfgs optimizer')
    parser.add_argument('--bsz', type=int, default=20,
                        help='training batch size')
    parser.add_argument('--l', type=int, default=13,
                        help='2**l+1 is total number of points')
    parser.add_argument('--k', type=int, default=5,
                        help='coarest level is l - k')
    parser.add_argument('--m', type=int, default=3,
                        help='neighborhood width')
    parser.add_argument('--ord', type=int, default=2,
                        help='interpolation order')
    
    args = parser.parse_args()

    ################################################################
    #  configurations
    ################################################################
    
    batch_size = args.bsz
    lr_adam = args.lr_adam
    lr_lbfgs = args.lr_lbfgs
    epochs_adam = args.epochs_adam
    epochs_lbfgs = args.epochs_lbfgs
    epochs = epochs_adam + epochs_lbfgs
    
    l = args.l
    k = args.k 
    m = args.m
    order = args.ord
    mlp_layers = [2, 50, 50, 1]

    now = datetime.now()
    exp_nm = now.strftime("exp-%Y-%m-%d-%H-%M-%S")
    device = torch.device(f'cuda:{args.device}')
    data_root = '/workdir/pde_data/green_learning/data1d_8193/'
    log_root = '/workdir/MLFormer/results/'
    task_nm = args.task
    model_nm = f'NMLK'
    hist_outpath, pred_outpath, model_operator_outpath, model_kernel_outpath, cfg_outpath = init_records(
        task_nm, log_root, model_nm, exp_nm)
    print('output files:')
    print(hist_outpath)
    print(pred_outpath)
    print(model_operator_outpath)
    print(model_kernel_outpath)
    print(cfg_outpath)

    with open(cfg_outpath, 'w') as f:
        cfg_dict = vars(args)
        cfg_dict['mlp'] = mlp_layers
        json.dump(cfg_dict, f)

    ################################################################
    # read data
    ################################################################
    train_loader, test_loader, Khh, w_hom, xh, grid_pts, h = load_dataset_1d(
        task_nm, data_root, bsz=batch_size)
    Khh = Khh.to(device)
    x = torch.tensor(xh).float().reshape(-1).to(device)
    x_nbrs = ((torch.arange(-m, m+1)/m + 1)/2).to(device)

    grid_pts = grid_pts.to(device)
    Khh = Khh.to(device)
    nmlk = NMLK(mlp_layers, l=l, k=k, m=m, order=order).to(device)
    nmlk.idx_j_lst = [x.to(device) for x in nmlk.idx_j_lst]

    ################################################################
    # training and evaluation
    ################################################################
    opt_adam = torch.optim.Adam(nmlk.parameters(), lr=lr_adam)
    opt_lbfgs = torch.optim.LBFGS(nmlk.parameters(), lr=lr_lbfgs)
    
    train_rl2_hist = []
    test_rl2_hist = []
    test_matrl2_hist = []
    train_rl2 = 1
    test_rl2_best = 1
    test_matrl2_best = 1
    pbar = trange(epochs)
    for ep in pbar:
        pbar.set_description("train l2 {:.2e} - test l2 {:.2e} - KernelNorm {:.2e}".format(
            train_rl2, test_rl2_best, test_matrl2_best))
        nmlk.train()
        train_rl2 = 0

        for u, w in train_loader:
            bsz = u.shape[0]
            u, w = u.to(device), w.to(device)
            rnd_rows = torch.randint(low=0, high=2**13+1, size=(16,)).to(device)
            w_rows = w[:,:,rnd_rows]

            if ep <= epochs_adam:
                nKHH, nKhh_banddiff_lst = nmlk(x, x_nbrs)
                nKhh = SingularSmoothKernelReconstruction(nKHH, nKhh_banddiff_lst, l, k, m)
                w_rows_ = multi_summation(nKhh[:,:,rnd_rows].repeat(bsz, 1, 1, 1), u, h)

                loss = rl2_error(w_rows_, w_rows)
                
                opt_adam.zero_grad()
                loss.backward()
                opt_adam.step()
            else:
                def loss_closure():
                    nKHH, nKhh_banddiff_lst = nmlk(x, x_nbrs)
                    nKhh = SingularSmoothKernelReconstruction(nKHH, nKhh_banddiff_lst, l, k, m)
                    w_rows_ = multi_summation(nKhh[:,:,rnd_rows].repeat(bsz, 1, 1, 1), u, h)
                    loss = rl2_error(w_rows_, w_rows)                    
                    opt_lbfgs.zero_grad()
                    loss.backward()
                    return loss 
                opt_lbfgs.step(loss_closure)
                nKHH, nKhh_banddiff_lst = nmlk(x, x_nbrs)
                nKhh = SingularSmoothKernelReconstruction(nKHH, nKhh_banddiff_lst, l, k, m)
                w_rows_ = multi_summation(nKhh[:,:,rnd_rows].repeat(bsz, 1, 1, 1), u, h)
                loss = loss_closure() 

            train_rl2 += loss.item()
        
        nmlk.eval()
        test_rl2 = 0
        with torch.no_grad():
            nKHH, nKhh_banddiff_lst = nmlk(x, x_nbrs)
            nKhh = SingularSmoothKernelReconstruction(nKHH, nKhh_banddiff_lst, l, k, m)
            
            for u, w in test_loader:
                bsz = u.shape[0]
                u, w = u.to(device), w.to(device)                
                w_ = multi_summation(nKhh.repeat(bsz,1,1,1), u, h)

                rl2 = rl2_error(w_, w)
                test_rl2 += rl2.item()

        test_matrl2 = matrl2_error(Khh, nKhh).item()
        train_rl2 = train_rl2/len(train_loader)
        test_rl2 = test_rl2/len(test_loader)

        train_rl2_hist.append(train_rl2)
        test_rl2_hist.append(test_rl2)
        test_matrl2_hist.append(test_matrl2)

        if test_rl2 < test_rl2_best:
            test_rl2_best = test_rl2
            torch.save(nmlk, model_operator_outpath)

        if test_matrl2 < test_matrl2_best:
            test_matrl2_best = test_matrl2
            torch.save(nmlk, model_kernel_outpath)

    save_hist(hist_outpath, train_rl2_hist, test_rl2_hist, test_matrl2_hist)
