"""
LSTM regression model for one watershed

Author: Abhinav Gupta (Created: 19 Apr 2022)

"""
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import copy

import torch
from torch import classes, nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, Sampler
from torchvision import datasets
from torchvision.transforms import ToTensor


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
    def __init__(self, X, y, seq_len):
        super(Dataset, self).__init__()
        self.X = X
        self.y = y
        self.L = seq_len    # sequence length
    
    def __len__(self):
        return len(self.y) - self.L
    
    def __getitem__(self, index):
        x1 = self.X[index - self.L+1 : index+1, :]
        y1 = self.y[index]
        return x1, y1

# custom loss function
class CustomLoss(nn.Module):
    def __init__(self, q):
        super(CustomLoss, self).__init__()
        self.q = q
        
    def forward(self, output, target):
        #loss = torch.mean(torch.square(output - target))
        res = output - target
        loss = (1-self.q)*(torch.sum(res[res>0])) - self.q*(torch.sum(res[res<0]))
        return loss

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
        self.dropout = nn.Dropout(p = 0.30)

    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim, device = x.device).requires_grad_()
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim, device = x.device).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0, c0))
        #out = self.fc(out[:,-1,:]) #use hn instead, use relu before fc
        # out = self.fc(self.dropout(hn[0,:,:]))
        out = self.fc(hn[0,:,:])
        #out=self.fc1(out)
        return out

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

# define the module to train the model
def train_mod(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    tr_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tr_loss += loss.item()    
        
    tr_loss /= size
    print(tr_loss)
    return tr_loss, model.state_dict()

# define the module to validate the model
def validate_mod(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= size
    print(f"Avg loss: {test_loss:>8f} \n")
    return test_loss

# define the module to test the model
def test_mod(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    sse = 0
    ynse = []
    pred_list = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            sse += torch.sum((pred - y)**2)
            ynse.append(y)
            pred_list.append(pred)
    ynse = torch.cat(ynse)
    pred_list = torch.cat(pred_list)
    sst = torch.sum((ynse - torch.mean(ynse))**2)
    nse = 1 - sse/sst
    test_loss /= num_batches
    print(f"Avg loss: {nse.item():>8f} \n")
    return nse.item(), pred_list, ynse

####################################################################################################
####################################################################################################
####################################################################################################
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
Lseq = 30 # sequence length
epochs = 50
hidden_dim = 128
n_layers = 1
output_dim = 1
nseeds = 8
lrate = 10**(-3)
#quant=0.9   # quantile to be predicted

# prepare data for LSTM regression
main_dir = 'D:/Research/datasets/data_generated/LSTM_data/MS01/prcp_daymet'
save_subdir = 'mean_predictions/LSTM_local_model_without_simSSC_ep_20_hdim_128_lr_0.001'

#plot_dir = 'D:/Research/plots/LSTM_local_models_without_MMF_calibration'
trn_frac= 0.60  # fraction of training data for calibration

fname = 'dataLSTM_without_simSSC_100_MS01.txt'
Xbeg = 1 # index to start picking predictor columns
Xlen = 5 # number of columsn to pick for predictor set 
fnameSplt = 'train_test_split.txt'

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
train_data, test_data, COMID_sr = [], [], []
for comid in np.unique(COMIDs):
    ind = np.nonzero(COMIDs==comid)[0]
    modtmp = model_data[ind,:]

    cm_ind = np.nonzero(trn_tst_splt[:,0]==comid)[0][0]
    datenum = trn_tst_splt[cm_ind, 1]

    trn_lst = np.nonzero(datenums_model==datenum)[0][0]
    train_data.append(modtmp[0:trn_lst+1,:])
    test_data.append(modtmp[trn_lst+1-Lseq+1:,:])
    COMID_sr.append(comid)

# prepare training data
for com_ind in range(0,len(COMID_sr)):

    trn = train_data[com_ind]
    tst = test_data[com_ind]

    # train set
    xtrain = trn[:,Xbeg:Xlen+1]
    ytrain = trn[:,0]
    train_inds = list(np.nonzero(np.isnan(ytrain)==False)[0])
    train_inds = [ii for ii in train_inds if ii>=Lseq-1]

    # test set
    xtest = tst[:,Xbeg:Xlen+1]
    ytest = tst[:,0]
    test_inds = list(np.nonzero(np.isnan(ytest)==False)[0])
    test_inds = [ind for ind in test_inds if ind>=Lseq-1]

    # convert data to tensor
    xtrain = torch.from_numpy(xtrain).float()
    ytrain = torch.from_numpy(ytrain.reshape(1,-1).T).float()

    xtest = torch.from_numpy(xtest).float()
    ytest = torch.from_numpy(ytest.reshape(1,-1).T).float()

    # normalize the data
    xmax, _ = torch.max(xtrain, 0)
    xmax[xmax==0] = 0.0001
    xtrain = torch.div(xtrain, xmax)
    xtest = torch.div(xtest, xmax)
    
    """
    meanx = torch.mean(xtrain, 0)
    sdx = torch.std(xtrain, 0)
    sdx[sdx==0] = 0.0001
    xtrain = torch.div(xtrain-meanx, sdx)
    xtest = torch.div(xtest-meanx, sdx)
    """

    ymax, _ = torch.max(ytrain[train_inds], 0)
    ytrain = torch.div(ytrain, ymax)
    ytest = torch.div(ytest, ymax)

    # create data loaders
    N = 2**5     # batch size
    train_dataset = CustomData(xtrain, ytrain, Lseq)
    test_dataset = CustomData(xtest, ytest, Lseq)
    trn_sampler = SubsetRandomSampler(train_inds)
    train_dataloader = DataLoader(train_dataset, batch_size = N, sampler=trn_sampler)
    
    g = torch.Generator()
    g.manual_seed(0)
    tst_sampler = SubsetSampler(test_inds)
    test_dataloader = DataLoader(test_dataset, batch_size = test_dataset.__len__(), sampler=tst_sampler)

    # Execute modeling for 8 random seeds 
    nse_ts_list = []
    ypred_list = []
    yobs_list = []
    for seed in range(nseeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # instantiate the model class
        input_dim = xtrain.shape[1]
        lstm = LSTMModel(input_dim, hidden_dim, n_layers, output_dim)
        lstm.cuda()

        # define loss and optimizer
        lossfn_train = nn.MSELoss()
        lossfn_test = nn.MSELoss()
        #lossfn_train = CustomLoss(quant)
        #lossfn_test = CustomLoss(quant)
        optimizer = torch.optim.Adam(lstm.parameters(), lr = lrate)

        # fix the number of epochs and start model training
        loss_tr_list = []
        loss_vl_list = []
        model_state = []
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            loss_tr, state = train_mod(train_dataloader, lstm, lossfn_train, optimizer)
            model_state.append(copy.deepcopy(state))
            loss_tr_list.append(loss_tr)
        lstm.load_state_dict(model_state[-1], strict = True)    
        nse_ts, ypred, yobs = test_mod(test_dataloader, lstm, lossfn_test)
        ypred = [ypred[ind][0].cpu().detach().numpy() for ind in range(len(ypred))]
        yobs = [yobs[ind][0].cpu().detach().numpy() for ind in range(len(yobs))]
        nse_ts_list.append(nse_ts)
        ypred_list.append(ypred)
        yobs_list.append(yobs)
        del lstm, optimizer, lossfn_train, lossfn_test
    
    ymax = ymax.cpu().detach().numpy()
    ypred = 0
    for y in ypred_list: ypred += np.array(y)
    ypred = ymax*ypred/nseeds
    yobs = ymax*np.array(yobs)
    nonanind = np.nonzero(np.isnan(yobs)==False)[0]
    ypred = ypred[nonanind]
    yobs = yobs[nonanind]
    nse = computeNSE(yobs, ypred)

    # save test streamflow data to a textfile along with nse of prediction on test set
    sname = 'obs_pred_' + str(COMID_sr[com_ind]) + '.txt'
    filename = os.path.join(main_dir, save_subdir, sname)
    fid = open(filename, 'w')
    fid.write('NSE = ' + str(nse) + '\n')
    fid.write('Observed\tPredicted\n')
    for wind in range(len(yobs)):
        fid.write('%f\t%f\n'%(yobs[wind], ypred[wind]))
    fid.close()

    """
    plt.scatter(yobs, ypred, s = 5)
    lim = np.max((np.max(yobs), np.max(ypred)))
    plt.ylim(0, lim)
    plt.xlim(0, lim)
    plt.plot([0, lim],[0, lim], color = 'black')
    plt.title('NSE = ' + str(round(nse*100)/100))
    plt.xlabel('Observed SSC')
    plt.ylabel('Predicted SSC')

    # save plot
    sname = str(COMID_sr[com_ind]) + '.tiff'
    filename = os.path.join(plot_dir, sname)
    plt.savefig(filename, dpi = 300)
    plt.close()
    """
    del train_dataset, test_dataset
    