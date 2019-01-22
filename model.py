import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class encoder(nn.Module):
    def __init__(self,num_person):
        super(encoder, self).__init__()
        # default image size :64
        self.conv1 = nn.Conv2d(3, 32, 3) # 3*64*64
        self.relu1 = nn.ReLU()
        self.max_pooling1=nn.MaxPool2d(2) # 32*32*32

        self.conv2 = nn.Conv2d(32, 64, 3)
        self.relu2 = nn.ReLU()
        self.max_pooling2 = nn.MaxPool2d(2)  # 64*16*16

        self.conv3 = nn.Conv2d(64, 128, 3)
        self.relu3 = nn.ReLU()
        self.max_pooling3 = nn.MaxPool2d(2)  # 128*8*8

        self.conv4 = nn.Conv2d(128, 256, 3)
        self.relu4 = nn.ReLU()
        self.max_pooling4 = nn.MaxPool2d(2)  # 256*4*4

        self.flatlayer = nn.MaxPool2d(4) # 256*1*1
        self.fc1 = nn.Linear(256,256)
        self.fc2 = nn. Linear(256,512)

        self.classifylayer = nn.Linear(512,num_person)
        self.softmax = nn.Softmax(num_person)
        ##TODO
        
    def forward(self, input):
        ##TODO
        x = self.conv1(input)
        x = self.relu1(x)
        x = self.max_pooling1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max_pooling2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.max_pooling3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.max_pooling4(x)

        x=self.flatlayer(x)
        x = x.view(-1,256)
        x=self.fc1(x)
        x=self.fc2(x)

        identity = self.softmax(self.classifylayer(x))

        return x, identity
    
class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        ##TODO

        self.fc1=nn.Linear(512,256)
        self.attributelayer = nn.Linear(6,256)
        # catenate
        # view(): # 256 *1*1
        self.relu0 = nn.ReLU()

        self.TransConv1 = nn.ConvTranspose2d(512,256,3,stride=4) # 256*4*4
        self.relu1 = nn.ReLU()

        self.TransConv2 = nn.ConvTranspose2d(256, 128, 3,stride=2) # 128*8*8
        self.relu2 = nn.ReLU()

        self.TransConv3 = nn.ConvTranspose2d(128, 64, 3,stride=2) # 64*16*16
        self.relu3 = nn.ReLU()

        self.TransConv4 = nn.ConvTranspose2d(64, 32, 3,stride=2) # 32*32*32
        self.relu4 = nn.ReLU()

        self.TransConv5 = nn.ConvTranspose2d(32, 3, 3,stride=2) # 3*64*64
        self.relu5 = nn.ReLU()


        
    def forward(self, content, attribute):
        ##TODO
        x = self.fc1(content)
        a = self.attributelayer(attribute)
        x = torch.cat(x,a)
        x = x.view(-1,256,1,1)

        x = self.relu0(x)

        x = self.relu1(x)
        x = self.TransConv1(x)

        x = self.relu2(x)
        x = self.TransConv2(x)

        x = self.relu3(x)
        x = self.TransConv3(x)

        x = self.relu4(x)
        x = self.TransConv4(x)

        x = self.relu5(x)
        x = self.TransConv5(x)
        return x
