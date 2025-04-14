"""

Benchmark LSTM model for prediction in time using data across several watersheds 

Author: Abhinav Gupta (Created: 2 May 2022)

"""
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from random import choices
import copy
import random
import gc

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, Sampler
from torchvision import datasets
from torchvision.transforms import ToTensor

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# function to compute NSE
def computeNSE(obs, pred):
    sse = np.sum((obs - pred)**2)
    sst = np.sum((obs - np.mean(obs))**2)
    nse = 1 - sse/sst
    return nse

# function to permute variables
def permute(trn, var_type, var_num):
    if var_type == 'dynamic':
        per_var = np.random.permutation(trn[:,var_num])
    trn[:,var_num]=  per_var
    return per_var


########################################################################################################
# LSTM classes
########################################################################################################
# build a dataset class
class CustomData(Dataset):
    def __init__(self, data, mean_X, std_X, stdy_list, data_inds, seq_len):
        super(Dataset, self).__init__()
        self.data = data
        self.L = seq_len            # sequence length
        self.mx = mean_X            # mean of predictor variables
        self.sdx = std_X            # standard deviation of predictor variables
        self.stdy_list = stdy_list  # maximum values of SSC for each COMID_ID
        self.index_map = data_inds  # indices to be used for model run (to exclude the indices related to NaN)
       
    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx):
        comid_ind, tind = idx
        x1 = self.data[comid_ind][tind - self.L+1 : tind+1, 1:]
        x1 = torch.div(x1 - self.mx, self.sdx)
        y1 = torch.div(self.data[comid_ind][tind,0], self.stdy_list[comid_ind])
        return comid_ind, x1, y1, self.stdy_list[comid_ind]

# custom loss function
class CustomLoss(nn.Module):
    def __init__(self, q):
        super(CustomLoss, self).__init__()
        self.q = q
        
    def forward(self, output, target):
        res = output-target
        loss = (1-self.q)*torch.sum(res[res>0]) - self.q*torch.sum(res[res<=0])
        return loss

# custom sampler function 
class SubsetSampler(Sampler):
    def __init__(self, indices, generator=None):
        super(Sampler, self).__init__()
        self.indices = indices
        self.generator = generator

    def __iter__(self):
        for i in self.indices:
            yield i

    def __len__(self) -> int:
        return len(self.indices)
    
# define LSTM model class
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first = True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = 0.40)

    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim, device = x.device).requires_grad_()
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim, device = x.device).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0, c0))
        #out = self.fc(out[:,-1,:]) #use hn instead, use relu before fc
        out = self.fc(self.dropout(hn[0,:,:]))
        #out = self.fc(hn[0,:,:])
        #out=self.fc1(out)
        return out

# define the module to train the model
def train_mod(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    
    tr_loss  = 0
    for batch, (comid_inds, X, y, stdy) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y.view(len(y),1))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tr_loss += loss.item()

        del loss, pred

    tr_loss /= size
    return tr_loss, model.state_dict()

# define the module to test the model
def test_mod(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    sse = 0
    ynse, pred_list, COMID_inds, stdy_out = [], [], [], []
    with torch.no_grad():
        for comid_inds, X, y, stdy in dataloader:
            X, y = X.to(device), y.to(device)
            y = y.view(len(y),1)
            pred = model(X)
            
            sse += torch.sum((pred - y)**2)
            ynse.append(y)
            pred_list.append(pred)
            COMID_inds.append(comid_inds)
            stdy_out.append(stdy)
    ynse = torch.cat(ynse)
    pred_list = torch.cat(pred_list)
    COMID_inds = torch.cat(COMID_inds)
    stdy_out = torch.cat(stdy_out)
    sst = torch.sum((ynse - torch.mean(ynse))**2)
    nse = 1 - sse/sst

    print(f"NSE: {nse.item():>8f} \n")
    return nse.item(), pred_list, ynse, COMID_inds, stdy_out

####################################################################################################
####################################################################################################
####################################################################################################
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
Lseq = 30
L = 30
epochs = 20
nseeds = 8
hidden_dim, n_layers, output_dim = 128, 1, 1
lrate = 10**(-3)

# prepare data for LSTM regression
main_dir = 'D:/Research/datasets/data_generated/LSTM_data/MS01/prcp_daymet'
save_subdir = 'mean_predictions/LSTMres_global_gauged_with_simSSC_Q_ep_{ep}_hdim_{hdim}_Lseq_{Lseq}_lr_{lr}'.format(ep=epochs, hdim=hidden_dim, Lseq=Lseq, lr=lrate)
trn_frac= 0.60  # fraction of training data for calibration

fname = 'dataLSTMres_with_simSSC_Q_100_MS01.txt'
fnameSplt = 'train_test_split.txt'

if os.path.exists(os.path.join(main_dir, save_subdir)) == False:
    os.mkdir(os.path.join(main_dir, save_subdir))

# read data
filename = os.path.join(main_dir, fname)
fid = open(filename, 'r')
data = fid.readlines()
fid.close()
COMIDs, datenums_model, model_data = [], [], []
for rind in range(1,len(data)):
    data_tmp = data[rind].split()
    COMIDs.append(int(float(data_tmp[0])))
    
    date_tmp = data_tmp[1]
    sp = date_tmp.split('-')
    yyyy, mm, dd = int(sp[0]), int(sp[1]), int(sp[2])
    datenums_model.append(datetime.date(yyyy, mm, dd).toordinal())

    model_data.append([float(x) for x in data_tmp[2:]])
COMIDs, datenums_model, model_data = np.array(COMIDs), np.array(datenums_model), np.array(model_data)

# read LSTM state dictionaries
state_list = []
for seed in range(nseeds):
    filename  = os.path.join(main_dir, save_subdir, 'LSTM_state{}'.format(seed))
    state = torch.load(open(filename, 'rb'))
    state_list.append(state)

# read train test split data
filename = os.path.join(main_dir, fnameSplt)
fid = open(filename, 'r')
data = fid.readlines()
fid.close()
trn_tst_splt = []
for rind in range(1, len(data)):
    data_tmp = data[rind].split()
    date_tmp = data_tmp[1]
    sp = date_tmp.split('-')
    yyyy, mm, dd = int(sp[0]), int(sp[1]), int(sp[2])
    trn_tst_splt.append([int(data_tmp[0]), datetime.date(yyyy, mm, dd).toordinal()])
trn_tst_splt = np.array(trn_tst_splt)

# create trainign and testing data
train_data, test_data, COMID_sr, stdy_list = [], [], [], []
for comid in np.unique(COMIDs):
    ind = np.nonzero(COMIDs==comid)[0]
    modtmp = model_data[ind,:]

    cm_ind = np.nonzero(trn_tst_splt[:,0]==comid)[0][0]
    datenum = trn_tst_splt[cm_ind, 1]

    trn_lst = np.nonzero(datenums_model==datenum)[0][0]
    
    trn_tmp = modtmp[0:trn_lst+1,:]
    stdy = np.nanstd(trn_tmp[:,0])


    train_data.append(trn_tmp)
    test_data.append(modtmp[trn_lst+1-L+1:,:])
    COMID_sr.append(comid)
    stdy_list.append(stdy)

stdy_list = np.array(stdy_list)
stdy_list = torch.from_numpy(stdy_list).float()

# compute mean and standard deviation of each column to be used for standardization
mean_data = np.vstack(train_data)
meanx = np.nanmean(mean_data[:, 1:], axis=0)
stdx = np.nanstd(mean_data[:,1:], axis=0)
stdx[stdx==0] = 0.001
meanx = torch.from_numpy(meanx).float()
stdx = torch.from_numpy(stdx).float()

# LSTM formatting of training and testing data
train_final, test_final, train_inds, test_inds = [], [], [], []
for com_ind in range(len(COMID_sr)):

    trn = train_data[com_ind]
    tst = test_data[com_ind]

    # train set
    ytrain = trn[:,0]
    trn_inds = list(np.nonzero(np.isnan(ytrain)==False)[0])
    trn_inds = [(com_ind,ii) for ii in trn_inds if ii>=L-1]

    # test set
    ytest = tst[:,0]
    tst_inds = list(np.nonzero(np.isnan(ytest)==False)[0])
    tst_inds = [(com_ind,ii) for ii in tst_inds if ii>=L-1]

    # convert data to tensor
    trn = torch.from_numpy(trn).float()
    tst = torch.from_numpy(tst).float()

    train_final.append(trn)
    test_final.append(tst)
    train_inds = train_inds + trn_inds
    test_inds = test_inds + tst_inds

# Randomly permute predictor variables and compute the NSE for each COMID after permutation
rng = np.random.default_rng(0)
nse_list = []
for var_num in range(0,test_final[0].shape[1]):
    test_final_per = copy.deepcopy(test_final)
    if var_num !=0: # if var_num!=0, then permute the variables
        for ii in range(len(test_final)):
            per_n = test_final_per[ii][:,var_num].shape[0]
            if var_num>7:   #if it is not a dynamic attribute
                per = rng.choice(mean_data[:,var_num], (per_n,), replace=True)
            else:
                per = rng.choice(test_final_per[ii][:,var_num], (per_n,), replace=True)
            test_final_per[ii][:,var_num] = torch.from_numpy(per).float()

    # create data loaders
    N = 2**5     # batch size
    test_dataset = CustomData(test_final_per, meanx, stdx, stdy_list, test_inds, Lseq)
    tst_sampler = SubsetSampler(test_inds)
    test_dataloader = DataLoader(test_dataset, batch_size = N, sampler=tst_sampler)

    input_dim = len(meanx)
    nse_ts_list, ypred_list, yobs_list = [], [], []
    for seed in range(nseeds):
        # instantiate the model class
        lstm = LSTMModel(input_dim, hidden_dim, n_layers, output_dim)
        lstm.cuda()
        lstm.load_state_dict(state_list[seed], strict = True)
        lossfn_test = nn.MSELoss()
        nse_ts, ypred, yobs, com_inds, stdy_out = test_mod(test_dataloader, lstm, lossfn_test)

        stdy_out = stdy_out.cpu().detach().numpy()
        ypred = [stdy_out[ind]*ypred[ind][0].cpu().detach().numpy() for ind in range(len(ypred))]
        yobs = [stdy_out[ind]*yobs[ind][0].cpu().detach().numpy() for ind in range(len(yobs))]
        com_inds = com_inds.cpu().detach().numpy()
        
        nse_ts_list.append(nse_ts)
        ypred_list.append(ypred)
        yobs_list.append(yobs)

        del lstm, lossfn_test

    # compute averages over the nseeds realizations
    # test set
    ypred = 0
    for y in ypred_list: ypred += np.array(y)
    ypred = ypred/nseeds
    yobs = np.array(yobs)
    com_inds = np.array(com_inds)
    nse = computeNSE(yobs, ypred)

    NSE = []
    for com_ind in np.unique(com_inds):
        ind = np.nonzero(com_inds==com_ind)[0]
        yobs_tmp, ypred_tmp, comid = yobs[ind], ypred[ind], COMID_sr[com_ind]
        nse = computeNSE(yobs_tmp, ypred_tmp)
        NSE.append([comid, nse])
    NSE = np.array(NSE)
    nse_list.append(NSE)

write_data = nse_list[0]
for wi in range(1, len(nse_list)):
   write_data = np.concatenate((write_data, nse_list[wi][:,1].reshape(1,-1).T), axis=1)

# save nse values after each permutation
sname = 'variable_importance_permutation_watershed_wise.txt'
np.savetxt(os.path.join(main_dir, save_subdir, sname), write_data)

