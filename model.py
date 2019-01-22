import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class encoder(nn.Module):
    def __init__(self,num_person):
        super(encoder, self).__init__()
        # default image size :64
        self.conv1 = nn.Conv2d(3, 32, 3) # 3*64*64
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 3)
        
        self.relu = nn.ReLU()
        self.max_pooling = nn.MaxPool2d(2) # 32*32*32

        self.flatlayer = nn.MaxPool2d(4) # 256*1*1
        self.fc1 = nn.Linear(256,256)
        self.fc2 = nn. Linear(256,512)

        self.classifylayer = nn.Linear(512,num_person)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(x)
        x = self.max_pooling(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pooling(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.max_pooling(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.max_pooling(x)

        x=self.flatlayer(x)
        x = x.view(-1,256)
        x=self.fc1(x)
        x=self.fc2(x)

        identity = self.softmax(self.classifylayer(x))

        return x, identity
    
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
