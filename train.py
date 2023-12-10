import torch 
from utils import rl2_error, ml_rl2_error, init_records, save_hist, save_preds
from mlmm import MLFormer
from dataset import load_dataset_1d
from tqdm import trange
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Train a mlformer operator learning in 1d")
    parser.add_argument('--device', type=int, default=0,
                        help='device id.')
    parser.add_argument('--task', type=str, default='cosine',
                        help='dataset name. (burgers, poisson, cosine, lnabs)')
    parser.add_argument('--mtype', type=str, default='lowrank',
                        help='global coarse model type')
    args = parser.parse_args()

    ################################################################
    #  configurations
    ################################################################
    
    batch_size = 20
    lr = 0.001
    epochs = 1000
    iterations = epochs*(1000//batch_size)
    
    l = 13 # number of level, total number of points is 2^l-1
    k = 3 # number of corase level
    m = 7 # local range for correction
    rank = 4 # rank of low-rank model
    order = 2 # order of interpolation/restriction

    device = torch.device(f'cuda:{args.device}')
    log_root = '/workdir/MLFormer/results/'
    task_nm = args.task
    model_nm = f'ML-{args.mtype}'
    mtype = args.mtype
    hist_outpath, pred_outpath, model_outpath = init_records(task_nm, log_root, model_nm)

    ################################################################
    # read data
    ################################################################
    upath = '/workdir/MLFormer/data/kernel_estimation/grf1d_N2000_s8193_m0_gamma1_tau1_sigma1_dirichlet.mat'
    wpath = f'/workdir/MLFormer/data/kernel_estimation/{task_nm}_8193.npy'
    train_loader, test_loader, Khh, xh, h = load_dataset_1d(task_nm, upath, wpath, bsz=batch_size)

    # build model 
    mlformer = MLFormer(rank=rank, m=m, k=k, l=l, h=h, order=order, mtype=mtype).to(device)

    ################################################################
    # training and evaluation
    ################################################################
    optimizer = torch.optim.Adam(mlformer.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

    train_rl2 = 1
    test_rl2_best = 1
    pbar = trange(epochs)
    for ep in pbar:
        pbar.set_description("train l2 {:.2e} - test l2 {:.2e}".format(train_rl2, test_rl2_best))
        mlformer.train()
        train_rl2 = 0
        for u, w in train_loader:
            u, w = u.to(device), w.to(device)
            optimizer.zero_grad()
            w_, _, _ = mlformer(u)
            loss = ml_rl2_error(w_, w, k, order)
            loss.backward()
            optimizer.step()
            train_rl2 += loss.item()
        # scheduler.step()
        
        mlformer.eval()
        test_rl2 = 0
        with torch.no_grad():
            for u, w in test_loader:
                u, w = u.to(device), w.to(device)
                w_, _, _ = mlformer(u)
                rl2 = rl2_error(w_[-1], w)
                test_rl2 += rl2.item()

        train_rl2 = train_rl2/len(train_loader)
        test_rl2 = test_rl2/len(test_loader)

        if test_rl2 < test_rl2_best:
            test_rl2_best = test_rl2
            torch.save(mlformer, model_outpath)

