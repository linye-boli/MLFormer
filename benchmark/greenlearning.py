import sys 
sys.path.append('../')

from einops import rearrange
import torch 
from utils import rl2_error, matrl2_error, init_records, save_hist, save_preds, multi_summation
from dataset import load_dataset_1d
from tqdm import trange
import argparse
from model import MLP, Rational

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Train green learning model in 1d")
    parser.add_argument('--device', type=int, default=0,
                        help='device id.')
    parser.add_argument('--task', type=str, default='cosine',
                        help='dataset name. (laplace, cosine, logarithm)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train.')
    args = parser.parse_args()

    ################################################################
    #  configurations
    ################################################################
    
    batch_size = 20
    lr_adam = 1e-3
    # lr_lbfgs = 1e-2/ # cosine/laplace
    lr_lbfgs = 1e-2 # laplace
    # logarithm
    
    epochs = args.epochs

    device = torch.device(f'cuda:{args.device}')
    # device = torch.device(f'cpu')
    data_root = '/workdir/pde_data/green_learning/data1d_8193/'
    log_root = '/workdir/MLFormer/results/'
    task_nm = args.task
    model_nm = f'GL'
    hist_outpath, pred_outpath, model_operator_outpath, model_kernel_outpath = init_records(task_nm, log_root, model_nm)
    print('output files:')
    print(hist_outpath)
    print(pred_outpath)
    print(model_operator_outpath)
    print(model_kernel_outpath)
    
    ################################################################
    # read data
    ################################################################
    train_loader, test_loader, Khh, w_hom, xh, grid_pts, h = load_dataset_1d(task_nm, data_root, bsz=batch_size)
    grid_pts = grid_pts.to(device)
    Khh = Khh.to(device)
    glnet = MLP([2, 50, 50, 50, 50, 1], Rational).to(device)

    ################################################################
    # training and evaluation
    ################################################################
    opt_adam = torch.optim.Adam(glnet.parameters(), lr=lr_adam)
    opt_lbfgs = torch.optim.LBFGS(glnet.parameters(), lr=lr_lbfgs)

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
        glnet.train()
        train_rl2 = 0
        
        for u, w in train_loader:
            bsz = u.shape[0]
            u, w = u.to(device), w.to(device)
            
            # since inference whole Khh is time consuming and memory limited,
            # random select rows for training            
            rnd_rows = torch.randint(low=0, high=2**13+1, size=(16,)).to(device)
            w_rows = w[:,:,rnd_rows]

            if ep < 400:
                Khh_rows = glnet(grid_pts[rnd_rows])
                Khh_rows = rearrange(Khh_rows, 'm n c->1 c m n').repeat(bsz, 1, 1, 1)
                w_rows_ = multi_summation(Khh_rows, u, h)
                loss = rl2_error(w_rows_, w_rows)
                opt_adam.zero_grad()
                loss.backward()
                opt_adam.step()
            else:
                # rnd_rows = torch.randint(low=0, high=2**13+1, size=(16,)).to(device)
                # w_rows = w[:,:,rnd_rows]
                def loss_closure():
                    Khh_rows = glnet(grid_pts[rnd_rows])
                    Khh_rows = rearrange(Khh_rows, 'm n c->1 c m n').repeat(bsz, 1, 1, 1)
                    w_rows_ = multi_summation(Khh_rows, u, h)
                    loss = rl2_error(w_rows_, w_rows)
                    opt_lbfgs.zero_grad()
                    loss.backward()
                    return loss
                # torch.nn.utils.clip_grad_norm_(glnet.parameters(), 0.1)
                opt_lbfgs.step(loss_closure)
                Khh_rows = glnet(grid_pts[rnd_rows])
                Khh_rows = rearrange(Khh_rows, 'm n c->1 c m n').repeat(bsz, 1, 1, 1)
                w_ = multi_summation(Khh_rows, u, h)
                loss = loss_closure() 

            train_rl2 += loss.item()

        glnet.eval()
        test_rl2 = 0
        with torch.no_grad():
            Khh_ = []
            for i in range(2**13+1):
                Khh_row = glnet(grid_pts[[i]])
                Khh_.append(Khh_row)
            
            Khh_ = torch.concat(Khh_)
            Khh_ = rearrange(Khh_, 'm n c -> 1 c m n')

            for u, w in test_loader.dataset:
                u, w = u.to(device), w.to(device)
                w_ = multi_summation(Khh_, u[None], h)
                test_rl2 += rl2_error(w_, w[None]).item()
        
        test_matrl2 = matrl2_error(Khh, Khh_).item()
        train_rl2 = train_rl2 / len(train_loader)
        test_rl2 = test_rl2 / len(test_loader.dataset)

        train_rl2_hist.append(train_rl2)
        test_rl2_hist.append(test_rl2)
        test_matrl2_hist.append(test_matrl2)

        if test_rl2 < test_rl2_best:
            test_rl2_best = test_rl2
            torch.save(glnet, model_operator_outpath)

        if test_matrl2 < test_matrl2_best:
            test_matrl2_best = test_matrl2
            torch.save(glnet, model_kernel_outpath)
        
    save_hist(hist_outpath, train_rl2_hist, test_rl2_hist, test_matrl2_hist)

    # preds = []
    # model = torch.load(model_outpath).to(device)
    # print('load model from : {:}'.format(model_outpath))
    # with torch.no_grad():
    #     Khh_ = []
    #     for i in range(2**13+1):
    #         Khh_row = glnet(grid_pts[[i]])
    #         Khh_.append(Khh_row)
        
    #     Khh_ = torch.concat(Khh_)
    #     Khh_ = rearrange(Khh_, 'm n c -> 1 c m n')

    #     for u, w in test_loader.dataset:
    #         u, w = u.to(device), w.to(device)
    #         w_ = multi_summation(Khh_, u[None], h)
    #         preds.append(w_[0].detach().cpu().numpy())
        
    # save_preds(pred_outpath, preds)
    