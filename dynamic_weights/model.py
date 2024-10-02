import torch.nn as nn
from pdb import set_trace as st
import torch
import torch.nn.functional as F

size_common_layers = 32
size_layer = 16
output_size = 1
class FCN_poro_as_one(nn.Module):
    def __init__(self,input_size):
        super(FCN_poro_as_one, self).__init__()

        self.layer_common = nn.Sequential(
            nn.Linear(input_size,size_common_layers),
            nn.Tanh(),
        )
        
        self.layer1 = nn.Sequential(
            nn.Linear(size_common_layers,size_layer),
            nn.Tanh(),
            nn.Linear(size_layer,output_size),
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(size_common_layers,size_layer),
            nn.Tanh(),
            nn.Linear(size_layer,output_size),
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(size_common_layers,size_layer),
            nn.Tanh(),
            nn.Linear(size_layer,output_size),
        )
        
        self.layer4 = nn.Sequential(
            nn.Linear(size_common_layers,size_layer),
            nn.Tanh(),
            nn.Linear(size_layer,output_size),
        )
        
        self.layer5 = nn.Sequential(
            nn.Linear(size_common_layers,size_layer),
            nn.Tanh(),
            nn.Linear(size_layer,output_size),
        )
        
        self.layer6 = nn.Sequential(
            nn.Linear(size_common_layers,size_layer),
            nn.Tanh(),
            nn.Linear(size_layer,output_size),
        )

    def forward(self, x):

        commonlayer = self.layer_common(x)
        out1 = self.layer1(commonlayer)
        out2 = self.layer2(commonlayer)
        out3 = self.layer3(commonlayer)
        out4 = self.layer4(commonlayer)
        out5 = self.layer5(commonlayer)
        out6 = self.layer6(commonlayer)

        ## network scaling factors theta defined from nominal value of each material parameter:
        theta1 = 1
        theta2 = 1
        theta3 = 1
        theta4 = 0.1
        theta5 = 1e-5
        theta6 = 1

        out1 = theta1*torch.abs(out1)
        out2 = theta2*torch.abs(out2)
        out3 = theta3*torch.abs(out3)    
        out4 = theta4*torch.abs(out4)
        out5 = theta5*torch.abs(out5)
        out6 = theta6*torch.abs(out6)



        return out1, out2, out3, out4, out5, out6
    

 