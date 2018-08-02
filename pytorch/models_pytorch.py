import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def move_data_to_gpu(x, cuda):

    if 'float' in str(x.dtype):
        x = torch.Tensor(x)

    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)

    else:
        raise Exception("Error!")

    if cuda:
        x = x.cuda()

    return x


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. 
    Ref: He, Kaiming, et al. "Delving deep into rectifiers: Surpassing 
    human-level performance on imagenet classification." Proceedings of the 
    IEEE international conference on computer vision. 2015.
    """
    
    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width
        
    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    
    bn.weight.data.fill_(1.)
    
    
class BaselineCnn(nn.Module):
    def __init__(self, classes_num):
        
        super(BaselineCnn, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64,
                               kernel_size=(5, 5), stride=(1, 2),
                               padding=(2, 2), bias=False)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=(5, 5), stride=(1, 2),
                               padding=(2, 2), bias=False)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=(5, 5), stride=(1, 2),
                               padding=(2, 2), bias=False)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=(5, 5), stride=(1, 2),
                               padding=(2, 2), bias=False)

        self.fc_final = nn.Linear(128, classes_num, bias=True)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.conv4)
        init_layer(self.fc_final)

        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)
        init_bn(self.bn4)

    def forward(self, input, return_frame_wise_probs=False):
        (_, seq_len, mel_bins) = input.shape

        x = input.view(-1, 1, seq_len, mel_bins)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        """(samples_num, feature_maps, time_steps, freq_num)"""

        (bottleneck, _) = torch.max(x, dim=-1)
        '''(samples_num, feature_maps, time_steps)'''

        x = torch.mean(bottleneck, dim=-1)
        '''(samples_num, feature_maps)'''
        
        output = F.sigmoid(self.fc_final(x))
        '''(samples_num, classes_num)'''

        if return_frame_wise_probs:
            
            mid_layer = bottleneck.transpose(1, 2)
            '''(samples_num, time_steps, feature_maps)'''
            
            frame_wise_prob = F.sigmoid(self.fc_final(mid_layer))
            '''(samples_num, time_steps, classes_num)'''
            
            return output, frame_wise_prob
        
        else:
            return output
            
            
class VggishConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(VggishConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.init_weights()
        
    def init_weights(self):
        
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        
    def forward(self, input):
        
        x = input
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=(1, 2), stride=(1, 2))
        
        return x
    
    
class Vggish(nn.Module):
    def __init__(self, classes_num):
        
        super(Vggish, self).__init__()

        self.conv_block1 = VggishConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = VggishConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = VggishConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = VggishConvBlock(in_channels=256, out_channels=512)

        self.fc_final = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc_final)

    def forward(self, input, return_frame_wise_probs=False):
        (_, seq_len, mel_bins) = input.shape

        x = input.view(-1, 1, seq_len, mel_bins)
        '''(samples_num, feature_maps, time_steps, freq_num)'''

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        
        (bottleneck, _) = torch.max(x, dim=-1)
        '''(samples_num, feature_maps, time_steps)'''

        # Averaging out time axis
        x = torch.mean(bottleneck, dim=-1)
        '''(samples_num, feature_maps)'''
        
        output = F.sigmoid(self.fc_final(x))
        '''(samples_num, classes_num)'''
        
        if return_frame_wise_probs:
            
            # Keep the time axis and apply dense layer later
            mid_layer = bottleneck.transpose(1, 2)
            '''(samples_num, time_steps, feature_maps)'''
            
            frame_wise_prob = F.sigmoid(self.fc_final(mid_layer))
            '''(samples_num, time_steps, classes_num)'''
            
            return output, frame_wise_prob
        
        else:
            return output