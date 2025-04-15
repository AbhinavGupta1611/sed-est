import torch
from models.lstm_model import computeNSE, CustomData, SubsetSampler, LSTMModel, train_mod, test_mod

def loadLSTM(filename, input_dim, hidden_dim, n_layers, output_dim):
    lstm = LSTMModel(input_dim, hidden_dim, n_layers, output_dim)
    lstm.cuda()
    state = torch.load(open(filename, 'rb'))
    return lstm.load_state_dict(state, strict = True)