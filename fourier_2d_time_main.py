"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Navier-Stokes equation discussed in Section 5.3 in the [paper](https://arxiv.org/pdf/2010.08895.pdf),
which uses a recurrent structure to propagates in time.
"""

#fourier_2d_time-main18-v72.py
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("fourier_neural_operator")

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


torch.manual_seed(0)
np.random.seed(0)
torch.manual_seed(42)
np.random.seed(42)

hyperparameter_defaults = dict(
    modes = 192,
    width = 5,
    epochs = 500,
    learning_rate = 0.001,
    scheduler_step = 100,
    scheduler_gamma = 0.5,
    batch_size = 1,
    tr_ratio = 0.8,
    std = 1,
    )

# Pass your defaults to wandb.init
wandb.init(config=hyperparameter_defaults, project="FNO_project", name='FNO-test1')
# Access all hyperparameter values through wandb.config
config = wandb.config

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
        self.xfft = x_ft
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


#dataset = loadmat('ion_200_300.mat')
#dataset = loadmat('DIR/beegfs/jeru889b-Smilei3/fno_1000.npy')

dataset = np.load('/bigdata/hplsim/aipp/Jeyhun/fno_main/fno_2000.npy')

print(dataset.shape,'dataset')
#x = dataset['x']
#px = dataset['px']

# Load the .npy file
# density = np.load('filename.npy')

#density = dataset['density']
density = np.transpose(dataset, (0, 2, 3, 1))
# d = dataset['d']
print(density.shape,'dataset after')
d_shape = density.shape[0]

# ntrain = 80
# ntest = 20
T_in = 1
T = 10 
NN = 2000

# ###### interpolation #########
tr_ratio = config["tr_ratio"]
total_index = np.arange(0,d_shape,1)
ntotal = d_shape
ntrain= int(np.round(d_shape*tr_ratio))
ntest = ntotal-ntrain
test_index = np.random.choice(NN, ntest, replace = False)
print('test_index',test_index)
print('test_index shape',test_index.shape)
train_index = np.setxor1d(test_index, total_index)
print('train_index.shape',train_index.shape)
# ###### interpolation #########

####### extrapolation ##########
# tr_ratio = config["tr_ratio"]
# total_index = np.arange(0,d.shape[1],1)
# ntotal = d.shape[1]
# ntrain= int(np.round(d.shape[1]*tr_ratio))
# ntest = ntotal-ntrain
# test_index = np.arange(ntrain,ntotal,1)
# print('test_index',test_index)
# print('test_index shape',test_index.shape)
# train_index = np.setxor1d(test_index, total_index)
# print('train_index.shape',train_index.shape)
####### extrapolation ##########


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

path = 'ns_fourier_ion_400_400_'+str(ntrain)+'_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width) + '_' + str(wandb.run.id)
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

# print('train_a.shape after', (train_a.shape))
# print('train_u.shape after', (train_u.shape))
# print('test_a.shape after', (test_a.shape))
# print('test_u.shape after', (test_u.shape))


train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

################################################################
# training and evaluation
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
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0
    for xx, yy in train_loader:
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)

        for t in range(0, T, step):
            y = yy[..., t:t + step]
            im = model(xx)
            
            loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step:], im), dim=-1)
            
        yy = y_normalizer.decode(yy)
        pred = y_normalizer.decode(pred)
        
        yy = torch.exp(yy) - 2
        pred = torch.exp(pred) - 2
        
        train_l2_step += loss.item()
        l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
        train_l2_full += l2_full.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    test_l2_step = 0
    test_l2_full = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)

            for t in range(0, T, step):
                y = yy[..., t:t + step]
                im = model(xx)
                                
                loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., step:], im), dim=-1)

            pred = y_normalizer.decode(pred)
            pred = torch.exp(pred) - 2
            
            test_l2_step += loss.item()
            test_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()
            
    # save best model        
    if min_valid_loss > test_l2_full:     
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{test_l2_full:.6f}) \t Saving The Model')
        min_valid_loss = test_l2_full
        # Saving State Dict
        torch.save(model.state_dict(), path_model, _use_new_zipfile_serialization=False)
    t2 = default_timer()
    scheduler.step()
    
    
#     print(ep, t2 - t1, train_l2_step / ntrain / (T / step), train_l2_full / ntrain, test_l2_step / ntest / (T / step), test_l2_full / ntest)
    
    print(f'Epoch {ep} \t time: {t2 - t1:.9f} \t train_l2_step: {train_l2_step / ntrain / (T / step):.9f} \t train_l2_full: {train_l2_full / ntrain:.9f} \t test_l2_step: {test_l2_step / ntest / (T / step):.9f} \t test_l2_full: {test_l2_full /ntest:.9f}')
    
    # Log the loss and accuracy values at the end of each epoch
    wandb.log({
        "Epoch": ep,
        "time": t2 - t1,
        "train_l2_steps": train_l2_step / ntrain / (T / step),
        "train_l2_full": train_l2_full / ntrain,
        "test_l2_step": test_l2_step / ntest / (T / step),
        "test_l2_full": test_l2_full / ntest,
        "min_valid_loss": min_valid_loss,
#         "weights real 0 w1": torch.nn.utils.parameters_to_vector(model.conv0.weights1).real,
#         "weights img 0 w1": torch.nn.utils.parameters_to_vector(model.conv0.weights1).imag,
#         "weights abs 0 w1": torch.nn.utils.parameters_to_vector(model.conv0.weights1).abs(),
        
#         "weights real 0 w2": torch.nn.utils.parameters_to_vector(model.conv0.weights2).real,
#         "weights img 0 w2": torch.nn.utils.parameters_to_vector(model.conv0.weights2).imag,
#         "weights abs 0 w2": torch.nn.utils.parameters_to_vector(model.conv0.weights2).abs(),
        
#         "weights real 1 w1": torch.nn.utils.parameters_to_vector(model.conv1.weights1).real,
#         "weights img 1 w1": torch.nn.utils.parameters_to_vector(model.conv1.weights1).imag,
#         "weights abs 1 w1": torch.nn.utils.parameters_to_vector(model.conv1.weights1).abs(),
        
#         "weights real 1 w2": torch.nn.utils.parameters_to_vector(model.conv1.weights2).real,
#         "weights img 1 w2": torch.nn.utils.parameters_to_vector(model.conv1.weights2).imag,
#         "weights abs 1 w2": torch.nn.utils.parameters_to_vector(model.conv1.weights2).abs(),
        
#         "weights real 2 w1": torch.nn.utils.parameters_to_vector(model.conv2.weights1).real,
#         "weights img 2 w1": torch.nn.utils.parameters_to_vector(model.conv2.weights1).imag,
#         "weights abs 2 w1": torch.nn.utils.parameters_to_vector(model.conv2.weights1).abs(),
        
#         "weights real 2 w2": torch.nn.utils.parameters_to_vector(model.conv2.weights2).real,
#         "weights img 2 w2": torch.nn.utils.parameters_to_vector(model.conv2.weights2).imag,
#         "weights abs 2 w2": torch.nn.utils.parameters_to_vector(model.conv2.weights2).abs(),
        
#         "weights real 3 w1": torch.nn.utils.parameters_to_vector(model.conv3.weights1).real,
#         "weights img 3 w1": torch.nn.utils.parameters_to_vector(model.conv3.weights1).imag,
#         "weights abs 3 w1": torch.nn.utils.parameters_to_vector(model.conv3.weights1).abs(),
        
#         "weights real 3 w2": torch.nn.utils.parameters_to_vector(model.conv3.weights2).real,
#         "weights img 3 w2": torch.nn.utils.parameters_to_vector(model.conv3.weights2).imag,
#         "weights abs 3 w2": torch.nn.utils.parameters_to_vector(model.conv3.weights2).abs(),
        
#         "x fft real": model.conv0.xfft.real,
#         "x fft imag": model.conv0.xfft.imag,
#         "x fft abs": model.conv0.xfft.abs()         
        })
    

# torch.save(model.state_dict(), path_model, _use_new_zipfile_serialization=False)

#Import the best model
model.load_state_dict(torch.load(path_model))
model.eval()

# Plots for test data
index = 0
prediction = torch.zeros(test_u.shape)
gt = torch.zeros(test_u.shape)
test_l2_step = 0
test_l2_full = 0
begin_test = time.time()
with torch.no_grad():
    for xx, yy in test_loader:
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)

        for t in range(0, T, step):
            y = yy[..., t:t + step]
            im = model(xx)
            
            loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)
            xx = torch.cat((xx[..., step:], im), dim=-1)

        ###me   
        pred = y_normalizer.decode(pred)
        pred = torch.exp(pred) - 2
        ###me
        
        test_l2_step += loss.item()
        test_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()
        
        prediction[index] = pred
        gt[index] = yy
        index = index + 1
end_test = time.time()       
test_time = end_test - begin_test
print("test time:", test_time)


# data = {'prediction': prediction.cpu().numpy(),
#         'gt': gt.cpu().numpy(),
#         }

# scipy.io.savemat('fno/pred_test/'+path+'.mat', data)




# # Plots for training data
# index = 0
# prediction_tr = torch.zeros(train_u.shape)
# gt_tr = torch.zeros(train_u.shape)
# # print('prediction_shape', prediction_tr.shape )
# # print('gt_shape', gt_tr.shape )
# test_l2_step = 0
# test_l2_full = 0
# begin_tr = time.time()
# with torch.no_grad():
#     for xx, yy in train_loader:
#         loss = 0
#         xx = xx.to(device)
#         yy = yy.to(device)
# #         print('xx.shape',xx.shape)
# #         print('yy.shape',yy.shape)

#         for t in range(0, T, step):
#             y = yy[..., t:t + step]
#             im = model(xx)
# #             print('imm shape',im.shape)
# #             #print('imm',im)
        
#             loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

#             if t == 0:
#                 pred = im
#             else:
#                 pred = torch.cat((pred, im), -1)
            
# #             print('xx[..., step:]', xx[..., step:].shape)
# #             print('im', im.shape)
# #             print('xx before after concat', xx.shape)
#             xx = torch.cat((xx[..., step:], im), dim=-1)
# #             print('xx shape after concat', xx.shape)

# #             print('pred', pred.shape)
            
        
#         ###me
#         yy = y_normalizer.decode(yy)
#         pred = y_normalizer.decode(pred)
#         ###mee
        
#         test_l2_step += loss.item()
# #         print('test_l2_step', test_l2_step)
#         test_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()
# #         print('test_l2_full',test_l2_full)
        
#         prediction_tr[index] = pred
#         gt_tr[index] = yy
#         index = index + 1
# end_tr = time.time()
# train_time = end_tr - begin_tr
# print("train_time:", train_time)

        
# data = {'prediction': prediction.cpu().numpy(),
#         'gt': gt.cpu().numpy(),
#         'prediction_tr': prediction_tr.cpu().numpy(),
#         'gt_tr': gt_tr.cpu().numpy(),
#         'test_time': test_time,
#         'train_time': train_time,
#         'test_index': test_index,
#         'train_index': train_index
#         }

        
data = {'prediction': prediction.cpu().numpy(),
        'gt': gt.cpu().numpy(),
        'test_time': test_time,
        'test_index': test_index,
        'train_index': train_index
        }



scipy.io.savemat('/bigdata/hplsim/aipp/Jeyhun/fno_main/fno_hemera/pred_data/'+path+'.mat', data)



# test_index = np.random.choice(100, ntest, replace = False)
# print('test_index',test_index)
# print('test_index shape',test_index.shape)
# train_index = np.setxor1d(test_index, total_index)

#print('prediction_tr', prediction_tr.shape)

# data = {'prediction_tr': prediction_tr.cpu().numpy(),
#         'gt_tr': gt_tr.cpu().numpy(),
#         }

# scipy.io.savemat('fno/pred_train/'+path+'.mat', data)


##########################################################################################################

# scipy.io.savemat('fno/pred/'+path+'.mat', format='5', mdict={'prediction': prediction.cpu().numpy(), 'gt': gt.cpu().numpy()})





# # torch.save(model, path_model)

# pred = torch.zeros(test_dataset.test_u.shape)
# index = 0
# #test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# y_normalizer = UnitGaussianNormalizer(Tensor(train_dataset.train_u).float())

# with torch.no_grad():
#     for x, y in test_loader:
#         test_l2 = 0;
#         x, y = x.cuda(), y.cuda()

#         out = model(x)
#         out = y_normalizer.decode(out)
#         pred[index] = out
        
#         print('y shape', y.shape)
#         print('out shape', out.shape)
#         test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
#         print(index, test_l2)
#         index = index + 1

# scipy.io.savemat('fno/pred/'+path+'.mat', mdict={'pred': pred.cpu().numpy(),
#                                             'y': y.cpu().numpy()})




