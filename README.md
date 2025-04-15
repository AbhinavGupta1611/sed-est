# sed-est
Codes to estimate suspended sediment concentration (SSC) using deep learning

Raw data and pre-trained models: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14902634.svg)](https://doi.org/10.5281/zenodo.14902634)

### Overview:

This repo contains codes to estimate SSC using several hydrometeorological and watershed static attribute data as inputs to the model. LSTM was used as the deep learning model and the models were developed in three scenarios: gauged, ungauged, and local. The repo can be used to reproduce the results in [Gupta and Feng (2025)](https://www.sciencedirect.com/science/article/pii/S0022169425004494).

### Installation Instructions

#### 1. Clone the repo
    git clone https://github.com/AbhinavGupta1611/sed-est.git

    cd sed-est

#### 2. Create a virtual environment
    python -m venv venv
    source venv/bin/activate     # On Windows use: venv\Scripts\activate

#### 3. Install required packages
    pip install -r requirements.txt

#### 4. Download the raw data and pretrained models from Zenodo  (Pre-trained models are required to run the only script scripts/predict.py) 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14902634.svg)](https://doi.org/10.5281/zenodo.14902634)
    
### Directory information
The repo contains both the standalone implementation and modular implemenation.

#### Standalone implementation 
**Mean_predictions_model** : Contains the standalone scripts that can be used to reproduce the results reproted in [Gupta and Feng (2025)](https://www.sciencedirect.com/science/article/pii/S0022169425004494).

#### Modular implementation
**models**: Contains the various pytorch classes required to run the model

**scripts**: contains various scripts to load the raw data, train LSTM, and load LSTM to predict SSC

### Usage example
The instruction below provides a guide to run a standalone script *LSTM_global_gauged.py*
1. Download the raw data from [Zenodo repository](https://zenodo.org/records/14902634) to a directory **sed_data**
2. Specify the LSTM hyperparameters. You can experiment with any hyperparameter comnbination you want. [Gupta and Feng (2025)](https://www.sciencedirect.com/science/article/pii/S0022169425004494) used the following hyperparameters combination:
   
        Lseq = 30       # Sequence length
        L = Lseq        # *L* is just a dummy variable which should be specified as equal to Lseq 
        epochs = 20     # Sequence length    
        nseeds = 8      # Number of models trained to deal with randomness in training
        hidden_dim, n_layers, output_dim = 256, 1, 1    # hidden dimension, number of layers, output dimension
        lrate = 10**(-3)        # Learning rate
        loss_fn = nn.L1Loss()   # Loss function
   
4. Specify the variable *main_dir* in which raw data are contained (assuming that raw data are saved in the directory D:/sed-data/raw_data)
5. Specify teh folder where trained models and predictions will be saved
6. Specify the name of the textfile containing raw data. There are five textfiles containing the raw data

   **dataLSTMres_without_simSSC_100_MS01.txt**: Use this file if you *do not want* to use MMF simulated SY as model input (This file will baseline model described in Gupta and Feng, 2025)

   **dataLSTMres_with_simSSC_100_MS01.txt**: Use this file if you *want* to use MMF simulated SY as model input

   **dataLSTMres_without_simSSC_Q_100_MS01.txt**: Use this file if you *do not want* to use MMF simulated SY as model input, but *want* to use VIC simulated streamflow as input

   **dataLSTMres_with_simSSC_Q_100_MS01.txt**: Use this file if you *want* to use both MMF simulated SY and VIC simulated streamflow as input

   **dataLSTMres_without_simSSC_Qobs_100_MS01.txt**: Use this file if you *do not want* to use MMF simulated SY as model input, but *want* to use observed streamflow as input

7. You can run the model now

An example code snippet that can be used to run the script:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    Lseq = 30
    L = Lseq
    epochs = 20
    nseeds = 8
    hidden_dim, n_layers, output_dim = 128, 1, 1
    lrate = 10**(-3)
    loss_fn = nn.L1Loss()
    #optim_lr = 0.7
    #use_RS = 'no'
    
    # prepare data for LSTM regression
    ############ Note: Learning rate scheduler was not used for 1st paper 
    main_dir = 'D:/sed-data/raw_data'
    save_subdir = 'LSTM_global_gauged_{}_{}_{}_{}'.format(Lseq, epochs, hidden_dim, lrate)
    if os.path.exists(os.path.join(main_dir, save_subdir)) == False:
        os.mkdir(os.path.join(main_dir, save_subdir))
    
    fname = 'dataLSTMres_without_simSSC_100_MS01.txt'
    fnameSplt = 'train_test_split.txt'

   

### References
Gupta, A., & Feng, D. (2025). Regional scale simulations of daily suspended sediment concentration at gauged and ungauged rivers using deep learning. Journal of Hydrology, 133111.
    


