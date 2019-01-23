import torch.nn as nn

def recon_loss(input, target):
    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()
    return mse_loss(input,target) + mae_loss(input,target)

def diff_loss(input, target):
    l1_loss = nn.L1Loss()
    if type(input) != list:
        print('z is not a list')
    return l1_loss(input[-1], target[-1])

def classify_loss(input, target):
    bce_loss = nn.BCELoss()
    return bce_loss(input, target)
 