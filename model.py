import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
class Encoder(nn.Module):
    def __init__(self,dim=64,n_layer=5):
        super(Encoder, self).__init__()
        # default image size :128
        d_in=3
        d_out=dim
        self.n_layer = n_layer
        self.blocks=[]
        for i in range(n_layer):
            conv = nn.Conv2d(d_in,d_out,4,2)
            bn = nn.BatchNorm2d(d_out)
            relu = nn.ReLU()
            seq=nn.Sequential(conv,bn,relu)
            self.blocks.append(seq)
            d_in=d_out
            d_out*=2
        self._weight_init()
        
    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()  
                
    def forward(self, input):
        EncoderOutputs=[]
        cur_input=input
        for i in range(self.n_layer):
            enc_out=self.blocks[i](cur_input)
            EncoderOutputs.append(enc_out)
            cur_input=enc_out
        return EncoderOutputs
    
class Decoder(nn.Module):
    def __init__(self, nlayers=5, shortcut_layer=1):
        super(Decoder, self).__init__()
        
        hidden_dim = [1064, 512, 256, 128, 64, 3]    
        self.layers = nlayers
        self.shortcut_layer = shortcut_layer
        self.blocks = []
        for i in range(nlayers-1):
            self.blocks.append(self.block(hidden_dim[i], hidden_dim[i+1]))
            
        self.blocks.append(nn.Sequential(nn.ConvTranspose2d(64,3,4,2),
                                     nn.BatchNorm2d(3),
                                     nn.Sigmoid()))
        self._weight_init()
        
    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def block(self, input_dim, output_dim):
        return nn.Sequential(nn.ConvTranspose2d(input_dim,output_dim,4,2),
                                     nn.BatchNorm2d(output_dim),
                                     nn.ReLU())
    
    def forward(self, z, attribute):  
        batch_size, att_num = attribute.shape
        att_expand = attribute.view(batch_size, att_num, 1, 1).expand_as(z[-1])
        x = torch.cat([z[-1], att_expand], dim=1)
        for i in range(self.layers):
            x = self.blocks[i](x)
            if i < self.shortcut_layer:
                x = x + z[i+1]
        return x
    
class Classifier(nn.Module):
    def __init__(self,num_classes=40,input_dimension=128,n_layer=5,dim=64):
        super(Classifier,self).__init__()
        self.num_classes=num_classes
        self.input_dimension=input_dimension
        self.n_layer=n_layer
        self.blocks = []
        d_in=3
        d_out=dim
        for i in range(n_layer):
            conv = nn.Conv2d(d_in, d_out, 4, 2)
            bn = nn.BatchNorm2d(d_out)
            relu = nn.ReLU()
            seq = nn.Sequential(conv, bn, relu)
            self.blocks.append(seq)
            d_in = d_out
            d_out *= 2

        # last conv layer :4*4*1024
        self.fc1=nn.Linear(4*4*1024,1024)
        self.fc2=nn.Linear(1024,self.num_classes)
        self.softmaxlayer = nn.Softmax(self.num_classes)
        self._weight_init()
        
    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.Linear):
                m.weight = nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, input):
        cur_input=input
        for i in range(self.n_layer):
            output = self.blocks[i](cur_input)
            cur_input = output

        output = output.view(-1,4*4*1024)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.softmaxlayer(output)

        return output
