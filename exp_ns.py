import os
import matplotlib.pyplot as plt
import argparse
import scipy.io as scio
import numpy as np
import torch
from tqdm import *
from utils.testloss import TestLoss
from model.PCSM_Structured_Mesh import Model

parser = argparse.ArgumentParser('PCSM')

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--n-hidden', type=int, default=64, help='hidden dim')
parser.add_argument('--n-layers', type=int, default=3, help='layers')
parser.add_argument('--n-heads', type=int, default=4)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument("--gpu", type=str, default='0', help="GPU index to use")
parser.add_argument('--max_grad_norm', type=float, default=None)
parser.add_argument('--downsample', type=int, default=1)
parser.add_argument('--mlp_ratio', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--unified_pos', type=int, default=0)
parser.add_argument('--ref', type=int, default=8)
parser.add_argument('--freq_num', type=int, default=32)
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--ntrain', type=int, default=1000)
parser.add_argument('--save_name', type=str, default='ns_2d_UniPDE')
parser.add_argument('--data_path', type=str, default='/data/fno')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
data_path = args.data_path + '/NavierStokes_V1e-5_N1200_T20.mat'
ntrain = args.ntrain
ntest = 200
T_in = 10
T = 10
step = 1
eval = args.eval
save_name = args.save_name  # + f'_DetIndex-{os.environ.get("DET_EXPERIMENT_ID")}'
print(f"Save Name: {save_name}")

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params

def main():
    r = args.downsample
    h = int(((64 - 1) / r) + 1)

    data = scio.loadmat(data_path)
    print(data['u'].shape)
    train_a = data['u'][:ntrain, ::r, ::r, :T_in][:, :h, :h, :]
    train_a = train_a.reshape(train_a.shape[0], -1, train_a.shape[-1])
    train_a = torch.from_numpy(train_a)
    train_u = data['u'][:ntrain, ::r, ::r, T_in:T + T_in][:, :h, :h, :]
    train_u = train_u.reshape(train_u.shape[0], -1, train_u.shape[-1])
    train_u = torch.from_numpy(train_u)

    test_a = data['u'][-ntest:, ::r, ::r, :T_in][:, :h, :h, :]
    test_a = test_a.reshape(test_a.shape[0], -1, test_a.shape[-1])
    test_a = torch.from_numpy(test_a)
    test_u = data['u'][-ntest:, ::r, ::r, T_in:T + T_in][:, :h, :h, :]
    test_u = test_u.reshape(test_u.shape[0], -1, test_u.shape[-1])
    test_u = torch.from_numpy(test_u)

    x = np.linspace(0, 1, h)
    y = np.linspace(0, 1, h)
    x, y = np.meshgrid(x, y)
    pos = np.c_[x.ravel(), y.ravel()]
    pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)
    pos_train = pos.repeat(ntrain, 1, 1)
    pos_test = pos.repeat(ntest, 1, 1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_train, train_a, train_u),
                                               batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_test, test_a, test_u),
                                              batch_size=args.batch_size, shuffle=False)

    print("Dataloading is over.")

    model = Model(space_dim=2,
                  n_layers=args.n_layers,
                  n_hidden=args.n_hidden,
                  dropout=args.dropout,
                  n_head=args.n_heads,
                  Time_Input=False,
                  mlp_ratio=args.mlp_ratio,
                  fun_dim=T_in,
                  out_dim=1,
                  freq_num=args.freq_num,
                  ref=args.ref,
                  unified_pos=args.unified_pos,
                  H=h, W=h).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(args)
    count_parameters(model)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,
                                                    steps_per_epoch=len(train_loader))
    myloss = TestLoss(size_average=False)

    for ep in tqdm(range(args.epochs)):

        model.train()
        train_l2_step = 0
        train_l2_full = 0

        for x, fx, yy in train_loader:
            loss = 0
            x, fx, yy = x.cuda(), fx.cuda(), yy.cuda()  # x: B,4096,2    fx: B,4096,T   y: B,4096,T
            bsz = x.shape[0]

            for t in range(0, T, step):
                y = yy[..., t:t + step]
                im = model(x, fx=fx)  # B , 4096 , 1
                loss += myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))
                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)
                fx = torch.cat((fx[..., step:], y), dim=-1)  # detach() & groundtruth

            train_l2_step += loss.item()
            train_l2_full += myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()
            optimizer.zero_grad()
            loss.backward()
            if args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

        test_l2_step = 0
        test_l2_full = 0

        model.eval()

        with torch.no_grad():
            for x, fx, yy in test_loader:
                loss = 0
                x, fx, yy = x.cuda(), fx.cuda(), yy.cuda()  # x : B, 4096, 2  fx : B, 4096  y : B, 4096, T
                bsz = x.shape[0]
                for t in range(0, T, step):
                    y = yy[..., t:t + step]
                    im = model(x, fx=fx)
                    loss += myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))
                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)
                    fx = torch.cat((fx[..., step:], im), dim=-1)

                test_l2_step += loss.item()
                test_l2_full += myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()

        print( "Epoch {} , train_step_loss:{:.5f} , train_full_loss:{:.5f} , test_step_loss:{:.5f} , test_full_loss:{:.5f}".format(
                ep, train_l2_step / ntrain / (T / step), train_l2_full / ntrain, test_l2_step / ntest / (T / step),
                test_l2_full / ntest))

        if ep % 100 == 0:
            if not os.path.exists('./checkpoints'):
                os.makedirs('./checkpoints')
            print('save model')
            torch.save(model.state_dict(), os.path.join('./checkpoints', save_name + '.pt'))

    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    print('save model')
    torch.save(model.state_dict(), os.path.join('./checkpoints', save_name + '.pt'))

if __name__ == "__main__":
    main()
