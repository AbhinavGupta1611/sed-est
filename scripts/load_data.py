import datetime
import os
import numpy as np
import torch

# Read raw data
def readRawData(main_dir, fname):
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
        
    return np.array(COMIDs), np.array(datenums_model), np.array(model_data)

# Read training and tetsing data
def TrainTestDates(main_dir, fnameSplt): 
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
    return np.array(trn_tst_splt)

# create trainign and testing data
def TrainingTestingData(COMIDs, model_data, trn_tst_splt, datenums_model, L):
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
    
    return train_data, test_data, COMID_sr, stdy_list

# LSTM formatting of training and testing data
def Tesorformatting(COMID_sr, train_data, test_data):
    train_final, test_final, train_inds, test_inds = [], [], [], []
    for com_ind in range(len(COMID_sr)):

        trn = train_data[com_ind]
        tst = test_data[com_ind]

        # Train set
        ytrain = trn[:,0]
        trn_inds = list(np.nonzero(np.isnan(ytrain)==False)[0])
        trn_inds = [(com_ind,ii) for ii in trn_inds if ii>=L-1]

        # Test set
        ytest = tst[:,0]
        tst_inds = list(np.nonzero(np.isnan(ytest)==False)[0])
        tst_inds = [(com_ind,ii) for ii in tst_inds if ii>=L-1]

        # Convert data to tensor
        trn = torch.from_numpy(trn).float()
        tst = torch.from_numpy(tst).float()

        train_final.append(trn)
        test_final.append(tst)
        train_inds = train_inds + trn_inds
        test_inds = test_inds + tst_inds
    
    return train_final, test_final, train_inds, test_inds