# HGCNSI-model
___
## Introduction of .py files
___
    'get_data.py' files are used to load data from public datasets and datasets collated from 'VGG19', and divide the data into 'train'/'test'/'val.pt' in the form of '.pt' for 'training', 'testing', and 'validation', respectively.
    
    metrics.py file provides the calculation method of the evaluation index (ADE/FDE), the calculation of Gaussian distribution loss and other related functions.
    
    model.py file provides the design details of the overall architecture of the model in this paper, including: the design of H-GCN, the design of SCA extracted features, and the construction of TCN.
    
    utils.py file provides the construction of hyperedges in hypergraphs and the construction of interaction modules between pedestrians.
    
    train.py file is used for model training.
    
    test.py file for testing purposes, where the particle filter module is used for single-track pedestrian trajectory prediction.
