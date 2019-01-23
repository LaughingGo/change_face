import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
class encoder(nn.Module):
    def __init__(self,dim=64,n_layer=5):
        super(encoder, self).__init__()
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
        
    def forward(self, input):
        EncoderOutputs=[]
        cur_input=input
        for i in range(self.n_layer):
            enc_out=self.blocks[i](cur_input)
            EncoderOutputs.append(enc_out)
            cur_input=enc_out
        return EncoderOutputs
    
class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()

        self.fc1=nn.Linear(512, 256)
        self.attribute_expand = nn.Linear(6, 256)

        self.TransConv1 = nn.ConvTranspose2d(512,256,3,stride=4) # 256*4*4
        self.TransConv2 = nn.ConvTranspose2d(256, 128, 3,stride=2) # 128*8*8
        self.TransConv3 = nn.ConvTranspose2d(128, 64, 3,stride=2) # 64*16*16
        self.TransConv4 = nn.ConvTranspose2d(64, 32, 3,stride=2) # 32*32*32
        self.TransConv5 = nn.ConvTranspose2d(32, 3, 3,stride=2,output_padding=1) # 3*64*64
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        
    def forward(self, content, attribute):                                                   
        x = self.fc1(content)
        a = self.attribute_expand(attribute)
        x = torch.cat([x,a],dim=1)
        x = x.view(-1,512,1,1)
        
        x = self.relu(self.TransConv1(x))
        x = self.relu(self.TransConv2(x))
        x = self.relu(self.TransConv3(x))
        x = self.relu(self.TransConv4(x))
        x = self.sigmoid(self.TransConv5(x))
        return x

    
class att_trans(nn.Module):
    def __init__(self,num_person):
        super(att_trans, self).__init__()
        self.encoder = encoder(num_person)
        self.decoder = decoder()
   
    def forward(self, img_1, img_att_2):
        content, identity = self.encoder(img_1)
        img2_pre = self.decoder(content, img_att_2)
        return img2_pre, identity

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
