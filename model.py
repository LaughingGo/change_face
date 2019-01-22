import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.conv1 = nn.Conv2d()
        self.relu = nn.ReLU()
        self.max_pooling = nn.MaxPooling()
        self.softmax = nn.SoftMax()
        ##TODO
        
    def forward(self, input):
        ##TODO
        return content, identity
    
class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.conv1 = nn.Conv2d()
        self.relu = nn.ReLU()
        self.max_pooling = nn.MaxPooling()
        self.softmax = nn.SoftMax()
        ##TODO
        
    def forward(self, content, attribute):
        ##TODO
        return pre_img
        