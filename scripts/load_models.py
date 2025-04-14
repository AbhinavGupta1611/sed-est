"""
script to load the models
"""

import torch
from lstm_model import LSTMModel

file_path = 'models/lstm_model.pth'  # Replace with actual path

# Choose input_dim:
# 19 → If MMF simulated SY and streamflow are NOT used
# 20 → If either MMF simulated SY OR streamflow is used
# 21 → If BOTH are used

input_dim = 21  # Change this based on your predictors
model = LSTMModel(input_dim=input_dim, hidden_dim=128, n_layers=1, output_dim=1)
model.load_state_dict(torch.load(file_path, map_location='cpu'))
model.eval()