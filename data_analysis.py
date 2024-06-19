import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("fourier_neural_operator")

from datetime import datetime
import matplotlib.pyplot as plt
from utilities3 import *

import operator
from functools import reduce
from functools import partial

from timeit import default_timer

from Adam import Adam

from torch.utils.data import Dataset, DataLoader
from torch import Tensor, ones, stack, load, nn
import wandb
from scipy.io import loadmat
from torch.nn.functional import normalize
import torch.fft
import time
import matplotlib.pyplot as plt
import os

torch.manual_seed(0)
np.random.seed(0)
torch.manual_seed(42)
np.random.seed(42)

def main(modes, width, tr_slice):
    config = dict(
        modes = 120,
        width = 5,
        epochs = 2,
        learning_rate = 0.001,
        scheduler_step = 100,
        scheduler_gamma = 0.5,
        batch_size = 1,
        tr_ratio = 0.8,
        std = 1,
        tr_slice = 100,
        model_checkpoint = 'rg8ne18c',
        save_plots = False,
        model_pathpattern = '/bigdata/hplsim/aipp/Jeyhun/fno_main/fno_hemera/fnomodel/ns_fourier_ion_400_400_{}_ep500_m{}_w{}_{}',
        plot_path = '/bigdata/hplsim/aipp/Jeyhun/fno_main/fno_hemera/fno_vs_interp/'
        )

    loaded_npzfile = np.load('/bigdata/hplsim/aipp/Jeyhun/fno_main/fno_hemera/models_dictionary.npz', allow_pickle=True)

    models_dictionary = {key: loaded_npzfile[key].item() for key in loaded_npzfile}

    # models_dictionary = {
    #     'rg8ne18c': {
    #         'modes': 120,
    #         'width': 5,
    #         'tr_slice': 100,
    #     },
    #     'kgwpsmos': {
    #         'modes': 120,
    #         'width': 5,
    #         'tr_slice': 200,
    #     },
    #     'f1zedebj': {
    #         'modes': 120,
    #         'width': 10,
    #         'tr_slice': 100,
    #     },
    #     'e5pk9vc0': {
    #         'modes': 120,
    #         'width': 10,
    #         'tr_slice': 200,
    #     },
    #     'h1mhpkze': {
    #         'modes': 120,
    #         'width': 20,
    #         'tr_slice': 100,
    #     },
    #     '5rv3hazi': {
    #         'modes': 120,
    #         'width': 20,
    #         'tr_slice': 200,
    #     },
    #     '4elfax8c': {
    #         'modes': 192,
    #         'width': 5,
    #         'tr_slice': 100,
    #     },
    #     'qdqhjqtl': {
    #         'modes': 192,
    #         'width': 5,
    #         'tr_slice': 200,
    #     },
    #     'ohyccl66': {
    #         'modes': 192,
    #         'width': 10,
    #         'tr_slice': 100,
    #     },
    #     'zbtscvpc': {
    #         'modes': 192,
    #         'width': 10,
    #         'tr_slice': 200,
    #     },
    #     '33892aeh': {
    #         'modes': 192,
    #         'width': 20,
    #         'tr_slice': 100,
    #     },
    #     '8lwt0yo7': {
    #         'modes': 192,
    #         'width': 20,
    #         'tr_slice': 200,
    #     },
    #     'yrlds00b': {
    #         'modes': 120,
    #         'width': 5,
    #         'tr_slice': 300,
    #     },
    #     'jgr8gvcb': {
    #         'modes': 120,
    #         'width': 5,
    #         'tr_slice': 400,
    #     },
    #     'stlemd8g': {
    #         'modes': 120,
    #         'width': 5,
    #         'tr_slice': 500,
    #     },
    #     'usrd9ri3': {
    #         'modes': 120,
    #         'width': 5,
    #         'tr_slice': None,
    #     },
    #     'nfkehgde': {
    #         'modes': 120,
    #         'width': 10,
    #         'tr_slice': 300,
    #     },
    #     '0qgk22jz': {
    #         'modes': 120,
    #         'width': 10,
    #         'tr_slice': 400,
    #     },
    #     'cqldfrgy': {
    #         'modes': 120,
    #         'width': 10,
    #         'tr_slice': 500,
    #     },
    #     '55gk8hh9': {
    #         'modes': 120,
    #         'width': 10,
    #         'tr_slice': None,
    #     },
    #     'uh8he2n8': {
    #         'modes': 120,
    #         'width': 20,
    #         'tr_slice': 300,
    #     },
    #     '8e2szlg4': {
    #         'modes': 120,
    #         'width': 20,
    #         'tr_slice': 400,
    #     },
    #     'wzuc9ty4': {
    #         'modes': 120,
    #         'width': 20,
    #         'tr_slice': 500,
    #     },
    #     'tkeh75fz': {
    #         'modes': 120,
    #         'width': 20,
    #         'tr_slice': None,
    #     },
    #     'zi9m2sta': {
    #         'modes': 192,
    #         'width': 5,
    #         'tr_slice': 300,
    #     },
    #     'izcte4dl': {
    #         'modes': 192,
    #         'width': 5,
    #         'tr_slice': 400,
    #     },
    #     'hurjr8b3': {
    #         'modes': 192,
    #         'width': 5,
    #         'tr_slice': 500,
    #     },
    #     'axu41mzi': {
    #         'modes': 192,
    #         'width': 10,
    #         'tr_slice': 300,
    #     },
    #     'gl2b58la': {
    #         'modes': 192,
    #         'width': 10,
    #         'tr_slice': 400,
    #     },
    #     't76rmuww': {
    #         'modes': 192,
    #         'width': 10,
    #         'tr_slice': 500,
    #     },
    #     '6pfad1bo': {
    #         'modes': 192,
    #         'width': 20,
    #         'tr_slice': 300,
    #     },
    #     'okpkzq6y': {
    #         'modes': 192,
    #         'width': 20,
    #         'tr_slice': 400,
    #     },
    #     'v8wpkh4x': {
    #         'modes': 192,
    #         'width': 20,
    #         'tr_slice': 500,
    #     },
    #     'le3dn2wt': {
    #         'modes': 192,
    #         'width': 5,
    #         'tr_slice': None,
    #     },
    #     'd7gwfh6k': {
    #         'modes': 192,
    #         'width': 10,
    #         'tr_slice': None,
    #     },
    #     'hl0oxyc4': {
    #         'modes': 192,
    #         'width': 20,
    #         'tr_slice': None,
    #     },
    # }

    def find_model(modes, width, tr_slice, data):
        for model_id, details in data.items():
            if details['modes'] == modes and details['width'] == width and details['tr_slice'] == tr_slice:
                return model_id
        return None

    checkpoint = find_model(modes, width, tr_slice, models_dictionary)

    if checkpoint:
        print(f"The checkpoint is identified as: {checkpoint}")
    else:
        print("No matching checkpoint found.")

    config["model_checkpoint"] = checkpoint
    config["modes"] = modes
    config["width"] = width
    config["tr_slice"] = tr_slice

    ################################################################
    # fourier layer
    ################################################################

    class SpectralConv2d_fast(nn.Module):
        def __init__(self, in_channels, out_channels, modes1, modes2):
            super(SpectralConv2d_fast, self).__init__()

            """
            2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
            """
            torch.manual_seed(42)
            np.random.seed(42)

            self.in_channels = in_channels
            self.out_channels = out_channels
            self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
            self.modes2 = modes2

            self.scale = (1 / (in_channels * out_channels))*config["std"]
            self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
            self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

        # Complex multiplication
        def compl_mul2d(self, input, weights):
            # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
            return torch.einsum("bixy,ioxy->boxy", input, weights)

        def forward(self, x):
            batchsize = x.shape[0]
            #Compute Fourier coeffcients up to factor of e^(- something constant)
            x_ft = torch.fft.rfft2(x)
            # self.xfft = x_ft
            # Multiply relevant Fourier modes
            out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
            out_ft[:, :, :self.modes1, :self.modes2] = \
                self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
            out_ft[:, :, -self.modes1:, :self.modes2] = \
                self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

            #Return to physical space
            x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
            return x

    class FNO2d(nn.Module):
        def __init__(self, modes1, modes2, width):
            super(FNO2d, self).__init__()

            """
            The overall network. It contains 4 layers of the Fourier layer.
            1. Lift the input to the desire channel dimension by self.fc0 .
            2. 4 layers of the integral operators u' = (W + K)(u).
                W defined by self.w; K defined by self.conv .
            3. Project from the channel space to the output space by self.fc1 and self.fc2 .

            input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
            input shape: (batchsize, x=64, y=64, c=12)
            output: the solution of the next timestep
            output shape: (batchsize, x=64, y=64, c=1)
            """

            self.modes1 = modes1
            self.modes2 = modes2
            self.width = width
            self.padding = 2 # pad the domain if input is non-periodic
            self.fc0 = nn.Linear(3, self.width)
            # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

            self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
            self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
            self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
            self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
            self.w0 = nn.Conv2d(self.width, self.width, 1)
            self.w1 = nn.Conv2d(self.width, self.width, 1)
            self.w2 = nn.Conv2d(self.width, self.width, 1)
            self.w3 = nn.Conv2d(self.width, self.width, 1)
            self.bn0 = torch.nn.BatchNorm2d(self.width)
            self.bn1 = torch.nn.BatchNorm2d(self.width)
            self.bn2 = torch.nn.BatchNorm2d(self.width)
            self.bn3 = torch.nn.BatchNorm2d(self.width)

            self.fc1 = nn.Linear(self.width, 128)
            self.fc2 = nn.Linear(128, 1)

        def forward(self, x):
            grid = self.get_grid(x.shape, x.device)
            x = torch.cat((x, grid), dim=-1)
            x = self.fc0(x)
            x = x.permute(0, 3, 1, 2)
            # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

            x1 = self.conv0(x)
            x2 = self.w0(x)
            x = x1 + x2
            x = F.gelu(x)

            x1 = self.conv1(x)
            x2 = self.w1(x)
            x = x1 + x2
            x = F.gelu(x)

            x1 = self.conv2(x)
            x2 = self.w2(x)
            x = x1 + x2
            x = F.gelu(x)

            x1 = self.conv3(x)
            x2 = self.w3(x)
            x = x1 + x2

            # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
            x = x.permute(0, 2, 3, 1)
            x = self.fc1(x)
            x = F.gelu(x)
            x = self.fc2(x)
            return x

        def get_grid(self, shape, device):
            batchsize, size_x, size_y = shape[0], shape[1], shape[2]
            gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
            gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
            gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
            gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
            return torch.cat((gridx, gridy), dim=-1).to(device)

    ################################################################
    # configs
    ################################################################

    d = np.load("/bigdata/hplsim/aipp/Jeyhun/fno_main/d_721.npy")
    print(d.shape)

    dataset = np.load('/bigdata/hplsim/aipp/Jeyhun/fno_main/fno_721.npy')
    print(dataset.shape,'dataset')

    density = np.transpose(dataset, (0, 2, 3, 1))
    # d = dataset['d']
    print(density.shape,'dataset after')
    ntotal = density.shape[0]

    T_in = 1
    T = 10 

    # ###### interpolation #########
    tr_ratio = config["tr_ratio"]
    total_index = np.arange(0,ntotal,1)
    ntrain= int(np.round(ntotal*tr_ratio))
    ntest = ntotal-ntrain
    test_index = np.random.choice(ntotal, ntest, replace = False)
    test_index = np.sort(test_index)
    print('test_index',test_index)
    print('test_index shape',test_index.shape)
    train_index = np.setxor1d(test_index, total_index)
    if config["tr_slice"]:
        ntrain = config["tr_slice"]
        train_index = np.random.choice(train_index, size=ntrain, replace=False)
    print('train_index.shape',train_index.shape)
    train_index = np.sort(train_index)

    dtest = d[test_index]
    dtrain = d[train_index]

    ## config update
    config.update({"test_index": test_index, "train_index": train_index, "total_index": total_index})
    ##

    modes = config["modes"]
    width = config["width"]
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    scheduler_step = config["scheduler_step"]
    scheduler_gamma = config["scheduler_gamma"]
    batch_size = config["batch_size"]

    print(epochs, learning_rate, scheduler_step, scheduler_gamma)

    path = 'ns_fourier_ion_400_400_'+str(ntrain)+'_ep' + str(epochs) + '_m' + str(modes) + '_w'# + str(width) + '_' + str(wandb.run.id)
    path_model = '/bigdata/hplsim/aipp/Jeyhun/fno_main/fno_hemera/fnomodel/'+path
    path_train_err = '/bigdata/hplsim/aipp/Jeyhun/fno_main/fno_hemera/fnoresults/'+path+'train.txt'
    path_test_err = '/bigdata/hplsim/aipp/Jeyhun/fno_main/fno_hemera/fnoresults/'+path+'test.txt'
    path_image = '/bigdata/hplsim/aipp/Jeyhun/fno_main/fno_hemera/fnoimage/'+path

    step = 1

    ################################################################
    # load data
    ################################################################

    train_a = Tensor(density[train_index,:,:,:T_in])
    train_u = Tensor(density[train_index,:,:,T_in:T+T_in])

    test_a = Tensor(density[test_index,:,:,:T_in])
    test_u = Tensor(density[test_index,:,:,T_in:T+T_in])

    # Replace zeros with -1
    train_a = torch.where(train_a == 0, torch.tensor(-1.), train_a)
    train_u = torch.where(train_u == 0, torch.tensor(-1.), train_u)
    test_a = torch.where(test_a == 0, torch.tensor(-1.), test_a)
    test_u = torch.where(test_u == 0, torch.tensor(-1.), test_u)

    # print('train_a.shape', (train_a.shape))
    # print('train_u.shape', (train_u.shape))
    # print('test_a.shape', (test_a.shape))
    # print('test_u.shape', (test_u.shape))

    # log transformation
    train_a = torch.log(train_a + 2)
    test_a = torch.log(test_a + 2)
    train_u = torch.log(train_u + 2)

    print('train_a shape', train_a.shape)

    #a_normalizer = UnitGaussianNormalizer(train_a)
    a_normalizer = GaussianNormalizer(train_a)
    #a_normalizer = RangeNormalizer(train_a)

    train_a = a_normalizer.encode(train_a)
    test_a = a_normalizer.encode(test_a)

    #y_normalizer = UnitGaussianNormalizer(train_u)
    y_normalizer = GaussianNormalizer(train_u)
    #y_normalizer = RangeNormalizer(train_u)

    train_u = y_normalizer.encode(train_u)


    print('a_normalizer.mean', a_normalizer.mean)
    print('a_normalizer.std', a_normalizer.std)

    print('y_normalizer.mean', y_normalizer.mean)
    print('y_normalizer.std', y_normalizer.std)

    train_index = torch.tensor(train_index)
    test_index = torch.tensor(test_index)

    train_dataset = torch.utils.data.TensorDataset(train_a, train_u, train_index)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False)

    test_dataset = torch.utils.data.TensorDataset(test_a, test_u, test_index)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False)


    ################################################################
    # evaluation
    ################################################################

    model = FNO2d(modes, modes, width).cuda()
    # model = torch.load('model/ns_fourier_V100_N1000_ep100_m8_w20')

    # #Log the network weight histograms (optional)
    # wandb.watch(model)

    print(count_params(model))
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    myloss = LpLoss(size_average=False)
    y_normalizer.cuda()
    min_valid_loss = np.inf

    path_model = config["model_pathpattern"].format(ntrain, config["modes"], config["width"], config["model_checkpoint"])

    print(path_model)

    begin_test = time.time()
    #Import the best model
    model.load_state_dict(torch.load(path_model))
    model.eval()
    end_test = time.time()       
    test_time = end_test - begin_test
    print(test_time)

    train_l2_step = 0
    train_l2_full = 0
    l2_full_all = torch.zeros(ntrain)
    index = 0
    with torch.no_grad():
        for xx, yy, train_id in train_loader:
            # print(index)
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)
            # print(train_id)
            for t in range(0, T, step):
                y = yy[..., t:t + step]
                im = model(xx)

                # loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., step:], im), dim=-1)

            yy = y_normalizer.decode(yy)
            pred = y_normalizer.decode(pred)

            yy = torch.exp(yy) - 2
            pred = torch.exp(pred) - 2

            # print(pred.shape)
            # reshaped = pred.reshape(-1, T)
            # reshaped2 = pred.reshape(batch_size, -1)
            # print(reshaped2.shape)

            # train_l2_step += loss.item()
            l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()
            l2_full_all[index] = l2_full
            # train_l2_full += l2_full.item()
            index +=1


    def find_training_neighbors(test_idx, train_indices):
        """Find the nearest training indices around a given test index."""
        pos = np.searchsorted(train_indices, test_idx)
        left = train_indices[pos-1] if pos > 0 else None
        right = train_indices[pos] if pos < len(train_indices) else None
        return left, right

    def linear_interpolation(x1, y1, x2, y2, x):
        return y1 + ((x - x1) * (y2 - y1) / (x2 - x1))


    test_l2_step = 0
    test_l2_full = 0
    test_l2_full_all = torch.zeros(ntest)
    test_interp_full_all = torch.zeros(ntest)

    index = 0
    with torch.no_grad():
        for xx, yy, test_id in test_loader:
            # print('index', index)
            # print('test_id', test_id)
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)

            # print('xx shape', xx.shape)

            for t in range(0, T, step):
                y = yy[..., t:t + step]
                im = model(xx)

                # loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., step:], im), dim=-1)

            # print('pred', pred.shape)
            pred = y_normalizer.decode(pred)
            pred = torch.exp(pred) - 2

            prev_id, next_id = find_training_neighbors(test_id, train_index)
            # print('test_id', test_id)
            # print('prev_id', prev_id)
            # print('next_id', next_id)

            if prev_id and next_id:
                train_index_list = train_index.tolist()
                load_prev_index = train_index_list.index(prev_id)
                load_next_index = train_index_list.index(next_id)

                # train_dataset = torch.utils.data.TensorDataset(train_a, train_u, train_index)

                # print('load_prev_index', load_prev_index)
                _,train_u_prev_id, assertion_prev_id = train_dataset[load_prev_index]

                assert prev_id == assertion_prev_id, f'Expected {prev_id} to be equal to {assertion_prev_id}'
                print("Assertion passed!")

                _,train_u_next_id, assertion_next_id = train_dataset[load_next_index]

                assert next_id == assertion_next_id, f'Expected {next_id} to be equal to {assertion_next_id}'
                print("Assertion passed!")

                d_prev_id = d[prev_id]
                d_next_id = d[next_id]
                d_test_id = d[test_id]

                train_u_prev_id = train_u_prev_id.unsqueeze(0).to(device)
                train_u_next_id = train_u_next_id.unsqueeze(0).to(device)
                # print('train_u_prev_id', train_u_prev_id.shape)

                begin_interp = time.time()
                interpolated_pred = linear_interpolation(x1=d_prev_id,
                                       y1=train_u_prev_id,
                                       x2=d_next_id,
                                       y2=train_u_next_id,
                                       x=d_test_id)
                end_interp = time.time()       
                interp_time = end_interp - begin_interp

                # interpolated_pred = interpolated_pred.unsqueeze(0)
                # print('interpolated_pred', interpolated_pred.shape)

                # yy = y_normalizer.decode(yy)
                interpolated_pred = y_normalizer.decode(interpolated_pred)
                # yy = torch.exp(yy) - 2
                interpolated_pred = torch.exp(interpolated_pred) - 2

                interp_full = myloss(interpolated_pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()
                test_interp_full_all[index] = interp_full

            # test_l2_step += loss.item()
            test_l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

            test_l2_full_all[index] = test_l2_full
            index +=1
    
    return l2_full_all, test_l2_full_all, test_interp_full_all, dtrain, dtest, checkpoint


if __name__ == "__main__":
    
    modes = [120, 192]
    lifting_dims = [5, 10, 20]
    train_sizes = [100,200,300,400,500,None]
    
    for mode in modes:
        print('mode', mode)
        for lifting_dim in lifting_dims:
            print('lifting_dim', lifting_dim)
            for train_size in train_sizes:
                print('train_size', train_size)
                l2_full_all, test_l2_full_all, test_interp_full_all, dtrain, dtest, checkpoint = main(mode, lifting_dim, train_size)
                
                results = {
                    'l2_full_all': l2_full_all,
                    'test_l2_full_all': test_l2_full_all,
                    'test_interp_full_all': test_interp_full_all,
                    'dtrain': dtrain,
                    'dtest': dtest,
                    'checkpoint': 'checkpoint',
                }
                
                file_name = "/bigdata/hplsim/aipp/Jeyhun/fno_main/fno_hemera/data_analysis/{}_{}_{}_{}.npz".format(mode,lifting_dim,train_size,checkpoint)
                np.savez(file_name, **results)
