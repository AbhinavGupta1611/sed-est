"""

Hold out validation for hyperparameter tuning
Author: Abhinav Gupta (Created: 13 Feb 2025)

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
Lseq_list = [10, 30, 60, 120, 180]
#L = 30
epochs_check = [5, 10, 20, 30, 40]
nseeds = 8
hidden_dims, n_layers, output_dim = [32], 1, 1
lrate_list = [10**(-3), 10**(-4)]
loss_fn = nn.MSELoss()

# prepare data for LSTM regressio
############ Note: Learning rate scheduler was not used for 1st paper 
main_dir = '/N/lustre/project/proj-212/abhinav/sediment/LSTM/LSTM_data/prcp_daymet'
#save_subdir = 'mean_predictions/LSTMres_global_gauged_43_L1Loss_simpler_without_simSSC_ep_{ep}_hdim_{hdim}_Lseq_{Lseq}_lr_{lr}'.format(ep=epochs, hdim=hidden_dim, Lseq=Lseq, lr=lrate)
save_subdir = 'mean_predictions/LSTMres_global_gauged_holdoutValidation_without_simSSC_Q'

trn_frac= 0.60  # fraction of training data for calibration

fname = 'dataLSTMres_without_simSSC_Q_100_MS01.txt'
fnameSplt = 'train_val_test_split.txt'

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

    # For SR data
    #model_data.append([float(x) for x in data_tmp[2:]]) # remote sensing data
    #model_data.append([float(x) for x in data_tmp[2:8]] + [float(x) for x in data_tmp[14:]]) # no remote sensing data
    #model_data.append([float(x) for x in data_tmp[2:7]] + [float(x) for x in data_tmp[14:16]]
    #                   + [float(x) for x in data_tmp[22:]])

    # without SR data
    model_data.append([float(x) for x in data_tmp[2:]])
    
COMIDs, datenums_model, model_data = np.array(COMIDs), np.array(datenums_model), np.array(model_data)

# randomly replace precipitation data with zeros
"""
np.random.seed(10000)
mis = np.random.choice([0.0,1.0], model_data.shape[0], p=binom_prob)
model_data[:,1] = model_data[:,1]*mis
"""

# read train test split data
filename = os.path.join(main_dir, fnameSplt)
fid = open(filename, 'r')
data = fid.readlines()
fid.close()
trn_val_tst_splt = []
for rind in range(1, len(data)):
    data_tmp = data[rind].split()
    
    # train_date
    date_tmp = data_tmp[1]
    sp = date_tmp.split('-')
    yyyy_trn, mm_trn, dd_trn = int(sp[0]), int(sp[1]), int(sp[2])

    # validation_date
    date_tmp = data_tmp[2]
    sp = date_tmp.split('-')
    yyyy_val, mm_val, dd_val = int(sp[0]), int(sp[1]), int(sp[2])

    trn_val_tst_splt.append([int(data_tmp[0]), datetime.date(yyyy_trn, mm_trn, dd_trn).toordinal(),
                             datetime.date(yyyy_val, mm_val, dd_val).toordinal()])
trn_val_tst_splt = np.array(trn_val_tst_splt)

# create trainign and validation data (validation data is denoted by 'test' in the script because of lagacy reasons )
train_data, test_data, COMID_sr, stdy_list = [], [], [], []
for comid in np.unique(COMIDs):
    ind = np.nonzero(COMIDs==comid)[0]
    modtmp = model_data[ind,:]
    
    # replace Nan in RS data by zeors
    """
    for ii in range(6,12):
        modtmp[np.isnan(modtmp[:,ii]),ii] = 0
    """
    
    cm_ind = np.nonzero(trn_val_tst_splt[:,0]==comid)[0][0]
    datenum_trn = trn_val_tst_splt[cm_ind, 1]
    datenum_val = trn_val_tst_splt[cm_ind, 2]

    trn_lst = np.nonzero(datenums_model==datenum_trn)[0][0]
    val_lst = np.nonzero(datenums_model==datenum_val)[0][0]
    
    trn_tmp = modtmp[0:trn_lst+1,:]
    stdy = np.nanstd(trn_tmp[:,0])


    train_data.append(trn_tmp)
    test_data.append(modtmp[trn_lst+1-np.max(Lseq_list)+1:val_lst+1,:])
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

############################################################################################################################################
# Model training

# create data loaders
N = 2**5     # batch size

for hidden_dim in hidden_dims:
    for lrate in lrate_list:
        for Lseq in Lseq_list:
            # LSTM formatting of training and testing data
            train_final, test_final, train_inds, test_inds = [], [], [], []
            for com_ind in range(len(COMID_sr)):

                trn = train_data[com_ind]
                tst = test_data[com_ind]

                # train set
                ytrain = trn[:,0]
                trn_inds = list(np.nonzero(np.isnan(ytrain)==False)[0])
                trn_inds = [(com_ind,ii) for ii in trn_inds if ii>=max(Lseq_list)-1]

                # test set
                ytest = tst[:,0]
                tst_inds = list(np.nonzero(np.isnan(ytest)==False)[0])
                tst_inds = [(com_ind,ii) for ii in tst_inds if ii>=max(Lseq_list)-1]

                # convert data to tensor
                trn = torch.from_numpy(trn).float()
                tst = torch.from_numpy(tst).float()

                train_final.append(trn)
                test_final.append(tst)
                train_inds = train_inds + trn_inds
                test_inds = test_inds + tst_inds

            train_dataset = CustomData(train_final, meanx, stdx, stdy_list, train_inds, Lseq)
            test_dataset = CustomData(test_final, meanx, stdx, stdy_list, test_inds, Lseq)
            trn_sampler = SubsetRandomSampler(train_inds)
            train_dataloader = DataLoader(train_dataset, batch_size = N, sampler=trn_sampler)


            tst_sampler = SubsetSampler(test_inds)
            test_dataloader = DataLoader(test_dataset, batch_size = N, sampler=tst_sampler)

            input_dim = len(meanx)
            nse_ts_list, ypred_list, yobs_list, test_COM_inds = [], [], [], []
            nse_sv1 = []
            for seed in range(nseeds):
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)

                # instantiate the model class
                lstm = LSTMModel(input_dim, hidden_dim, n_layers, output_dim)
                lstm.cuda()

                # define loss and optimizer
                lossfn_train = loss_fn
                #lossfn_train = CustomLoss(quant)
                optimizer = torch.optim.Adam(lstm.parameters(), lr = lrate)

                # fix the number of epochs and start model training
                loss_tr_list = []
                loss_vl_list = []
                model_state = []
                for t in range(np.max(epochs_check)):
                    print(f"Epoch {t+1}\n-------------------------------")
                    loss_tr, state = train_mod(train_dataloader, lstm, lossfn_train, optimizer)
                    model_state.append(copy.deepcopy(state))
                    loss_tr_list.append(loss_tr)
                    torch.cuda.empty_cache()
                
                # save obs and pred values for each of the epcohs in epoch_check
                for ep in epochs_check:
                    lstm = LSTMModel(input_dim, hidden_dim, n_layers, output_dim)
                    lstm.cuda()
                    lstm.load_state_dict(model_state[ep-1], strict = True)
                    lossfn_test = loss_fn
                    nse_ts, ypred, yobs, com_inds, stdy_out = test_mod(test_dataloader, lstm, lossfn_test)

                    stdy_out = stdy_out.cpu().detach().numpy()
                    ypred = [stdy_out[ind]*ypred[ind][0].cpu().detach().numpy() for ind in range(len(ypred))]
                    yobs = [stdy_out[ind]*yobs[ind][0].cpu().detach().numpy() for ind in range(len(yobs))]
                    com_inds = com_inds.cpu().detach().numpy()

                    del lstm

                    # write data to textfiles
                    tmp_dir = 'obs_pred_ep_{}_Lseq_{}_hdim_{}_lrate_{}_seed_{}'.format(ep, Lseq, hidden_dim, lrate, seed)
                    save_dir = os.path.join(main_dir, save_subdir, tmp_dir)
                    if os.path.exists(save_dir)==False:
                        os.mkdir(save_dir)
                    NSE = []
                    yobs, ypred = np.array(yobs), np.array(ypred)
                    for com_ind in np.unique(com_inds):
                        ind = np.nonzero(com_inds==com_ind)[0]
                        yobs_tmp, ypred_tmp, comid = yobs[ind], ypred[ind], COMID_sr[com_ind]
                        nse = computeNSE(yobs_tmp, ypred_tmp)
                        NSE.append([comid, nse])
                        
                        sname = 'obs_pred_{}.txt'.format(comid)
                        filename = os.path.join(save_dir, sname)
                        fid = open(filename, 'w')
                        fid.write('NSE = ' + str(nse) + '\n')
                        fid.write('Observed\tPredicted\n')
                        for wind in range(len(yobs_tmp)):
                            fid.write('%f\t%f\n'%(yobs_tmp[wind], ypred_tmp[wind]))
                        fid.close()
                del optimizer, lossfn_train, lossfn_test
#############################################################################################################################################
