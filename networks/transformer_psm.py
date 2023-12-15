# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch 
import torch.nn
import torchvision
from torch.autograd import Variable
from networks.psm_submodules import *
from torchvision.models import vit_b_16
from torchvision.models import resnet18
from torchvision.models.vision_transformer import ViT_B_16_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names
import torch.nn.functional as F
from pprint import pprint
from torchinfo import summary


def print_model_details(model_name : str):
    """
    Prints the node details of the torchvision model.
    """
    network = getattr(torchvision.models, model_name)(pretrained=True)
    train_nodes, eval_nodes = get_graph_node_names(network)
    pprint(train_nodes)   



class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1
    
    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x

class AddReadout(nn.Module):
    def __init__(self, start_index = 1):
        super(AddReadout, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        if self.start_index == 2:
            readout = (x[:, 0] + x[:, 1]) / 2
        else:
            readout = x[:, 0]
        return x[:, self.start_index : ] + readout.unsqueeze(1)


class StereoImageModel(nn.Module):

    def __init__(self):
        super(StereoImageModel, self).__init__()
        
        self.drop_out = 0.1
        self.max_disparity = 192
        self.read_out = AddReadout()
        self.size = [224, 224]

        # SPP
        self.Spp_module = feature_extraction(big_SPP= False)
        self.imageupsample = nn.ConvTranspose2d(in_channels=3, out_channels= 32, kernel_size= 2, stride= 2)

        # ViT Encoder.
        self.vit_encoder = vit_b_16(weights= ViT_B_16_Weights.DEFAULT)
        self.feature_extraction = create_feature_extractor(self.vit_encoder, return_nodes=['encoder'])
        
        # Standard transformer info
        self.feature_size = 768
        self.patch_size = 16
        self.num_patches = 14

        # Cross Attention block: 
        self.cross_atten = nn.MultiheadAttention(embed_dim= self.feature_size, num_heads= 16, batch_first= True)
        self.layer_norm1 = nn.LayerNorm(self.feature_size)
        self.layer_norm2 = nn.LayerNorm(self.feature_size)
        self.feedforward = nn.Sequential(
            nn.Linear(self.feature_size, 4 * self.feature_size),
            nn.ReLU(),
            nn.Linear(4 * self.feature_size, self.feature_size),
            nn.Dropout(self.drop_out)
        )

        # Reconstruction of the tokens 
        self.reconstruction_layer = nn.Sequential(
            self.read_out,
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([self.size[0] // 16, self.size[1] // 16])),
            nn.Conv2d(
                in_channels=self.feature_size,
                out_channels= 128,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels= 128,
                out_channels= 64,
                kernel_size=4,
                stride=4,
                padding=0,
                bias=True,
                dilation=1,
                groups=1,
            ),
            nn.ConvTranspose2d(
                in_channels= 64,
                out_channels= 32,
                kernel_size=4,
                stride=4,
                padding=0,
                bias=True,
                dilation=1,
                groups=1,
            ),
        )


        #Resnet based CNNs
        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn_3d(32, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 

        self.dres2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))
 
        self.dres3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 

        self.dres4 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 
 
        self.classify = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))



    def forward(self, x1, x2):
        output = {}
        # Store the original image size.
        b, c, h , w = x1.shape

        # Get the ViT encoded images : 
        left = self.feature_extraction(x1)['encoder']
        right = self.feature_extraction(x2)['encoder']
        
        #Cross attention layer : 
        left_ = left.clone()
        right_ = right.clone()
        
        left_encoding, _ = self.cross_atten(query= left_, key= right_, value= right_)
        left = self.layer_norm1(left + left_encoding)
        feed_forward = self.feedforward(left)
        left = self.layer_norm2(left + feed_forward)

        right_encoding, _ = self.cross_atten(query= right_, key= left_, value= left_)
        right = self.layer_norm1(right + right_encoding)
        feed_forward = self.feedforward(right)
        right = self.layer_norm2(right + feed_forward)

        del left_
        del right_

        # Reconstruction block 
        left = self.reconstruction_layer(left)
        right = self.reconstruction_layer(right)

        channel = left.size()[1] 
        
        # Cost volume : 
        cost = Variable(
            torch.FloatTensor(b, channel * 2, int(self.max_disparity / 4), left.size()[2] , 
            left.size()[3]).zero_()).cuda()

        for i in range(int(self.max_disparity / 4 )):
            if i > 0 :
                cost[:, :channel, i, :, i:] = left[:, :, :, i:]
                cost[:, channel:, i, :, i:] = right[:, :, :, :-i]

            else:
                cost[:, :channel, i, :, :] = left
                cost[:, channel:, i, :, :] = right
        cost = cost.contiguous()


        # Pass through the resnet based cnns. 
        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0
        cost0 = self.dres2(cost0) + cost0 
        cost0 = self.dres3(cost0) + cost0 
        cost0 = self.dres4(cost0) + cost0

        cost = self.classify(cost0)
        cost = F.interpolate(cost, size=[self.max_disparity, h, w], mode='trilinear', align_corners=False)
        cost = torch.squeeze(cost, 1)
        pred = F.softmax(cost, dim= 1)
        pred = disparityregression(self.max_disparity)(pred)
        
        output[('raw', 0)] = pred.unsqueeze(1)
        return output
