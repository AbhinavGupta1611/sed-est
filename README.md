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

#### 4. Download the pretrained models from Zenodo (Optional: required to run the script scripts/predict.py)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14902634.svg)](https://doi.org/10.5281/zenodo.14902634)
    
    
### References
Gupta, A., & Feng, D. (2025). Regional scale simulations of daily suspended sediment concentration at gauged and ungauged rivers using deep learning. Journal of Hydrology, 133111.
    


