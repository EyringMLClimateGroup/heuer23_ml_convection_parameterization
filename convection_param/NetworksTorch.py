import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyMaxPool1D(nn.Module):
    '''MaxPool1D layer that works with shap, only works with data in format BCH'''
    def __init__(self, kernel_size, **args):
        super(MyMaxPool1D, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=(kernel_size,1), **args)

    def forward(self, x):
        x = torch.reshape(x, (-1,x.shape[1],x.shape[2],1))
        x = self.pool(x)
        x = torch.squeeze(x, -1)
        return x

#-------------------------------------------------------------------------------------#
##################################### Sequential #####################################
#-------------------------------------------------------------------------------------#

class Sequential(nn.Module):
    def __init__(self, input_dim, output_dim, n_hidden, n_layers, activation=F.relu, bn=False):
        super(Sequential, self).__init__()
        self.activation = activation
        self.fc_in = nn.Linear(input_dim, n_hidden)
        self.fcs = nn.ModuleList([nn.Linear(n_hidden, n_hidden) for _ in range(n_layers-1)])
        if bn:
            self.bns = nn.ModuleList([nn.BatchNorm1d(n_hidden) for _ in range(n_layers-1)])
        else:
            self.bns = nn.ModuleList([nn.Identity() for _ in range(n_layers)])
        self.fc_out = nn.Linear(n_hidden, output_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.activation(self.fc_in(x))
        for fc,bn in zip(self.fcs, self.bns):
            x = self.activation(bn(fc(x)))
        x = self.fc_out(x)
        return x

#-------------------------------------------------------------------------------------#
##################################### Convolution #####################################
#-------------------------------------------------------------------------------------#

class SeqConv(nn.Module):
    def __init__(self, n_channels, n_feature_channels, column_height, n_hidden, n_layers, output_dim, kernel_size, activation=F.relu):
        super(SeqConv, self).__init__()
        self.activation = activation
        self.conv1 = nn.Conv1d(n_channels, n_feature_channels, kernel_size=kernel_size)
        self.fc1 = nn.Linear(n_feature_channels*(column_height-kernel_size+1), n_hidden)
        fcs = [nn.Linear(n_hidden, n_hidden) for _ in range(n_layers)]
        self.fcs = nn.ModuleList(fcs)
        self.fc_final = nn.Linear(n_hidden, output_dim)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = torch.flatten(x, 1)
        x = self.activation(self.fc1(x))
        for fc in self.fcs:
            x = self.activation(fc(x))
        x = self.fc_final(x)
        return x


#------------------------------------------------------------------------------------------------#
####################################### Unet Components #######################################
#------------------------------------------------------------------------------------------------#


class DoubleConv(nn.Module):
    """conv -> bn -> act -> conv -> bn -> act"""
    def __init__(self, in_channels, out_channels, middle_channels=None, bn1=True, bn2=True, activation=F.relu):
        super().__init__()
        self.activation = activation
        if not middle_channels:
            middle_channels = out_channels
        self.conv1 = nn.Conv1d(in_channels, middle_channels, kernel_size=3, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm1d(middle_channels) if bn1 else nn.Identity()
        self.bn1 = nn.BatchNorm1d(middle_channels)
        self.conv2 = nn.Conv1d(middle_channels, out_channels, kernel_size=3, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm1d(out_channels) if bn2 else nn.Identity()
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        return x


class Down(nn.Module):
    """Downscaling -> double conv"""
    def __init__(self, in_channels, out_channels, bn1=True, bn2=True, activation=F.relu):
        super().__init__()
        self.maxpool_conv = nn.Sequential(MyMaxPool1D(2), DoubleConv(in_channels, out_channels, bn1=bn1, bn2=bn2, activation=activation))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling -> double conv"""
    def __init__(self, in_channels, out_channels, linear=True, bn1=True, bn2=True, activation=F.relu):
        super().__init__()
        if linear:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, bn1=bn1, bn2=bn2, activation=activation)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, bn1=bn1, bn2=bn2, activation=activation)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is of form CH
        diffH = x2.size()[2] - x1.size()[2]

        x1 = F.pad(x1, [diffH // 2, diffH - diffH // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final convolution to get output with desired number of channels"""
    def __init__(self, in_channels, out_channels, activation=F.relu):
        super(OutConv, self).__init__()
        self.activation = activation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.activation(self.conv(x))

class PadLastDim(nn.Module):
    """Padd the last dimension of input tensor to some desired size"""
    def __init__(self, size):
        super(PadLastDim, self).__init__()
        self.size = size

    def forward(self, x):
        diff = self.size - x.shape[-1]
        x = F.pad(x, [diff // 2, diff - diff // 2])
        return x

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)


#--------------------------------------------------------------------------------------#
####################################### Unet #######################################
#--------------------------------------------------------------------------------------#


def get_pad_last_dim(diff):
    def pad_last_dim(x):
        return F.pad(x, [diff//2, diff - diff//2])
    #TODO: switch to this and try to pickle instead of lambda
    return pad_last_dim

class Unet(nn.Module):
    def __init__(self, n_channels, n_classes, output_channels_total, n_levels, n_features, column_height=23, linear=False, bn1=True, bn2=True, activation=F.relu):
        super(Unet, self).__init__()
        factor = 2 if linear else 1

        # Unet can only handle heights as power of 2
        transformed_column_height = int(2**np.ceil(np.log2(column_height)))
        diff = transformed_column_height - column_height
        self.init_transform = lambda x: F.pad(x, [diff//2, diff - diff//2]) # Shorthand for PadLastDim
        # self.init_transform = LambdaLayer(lambda x: F.pad(x, [diff//2, diff - diff//2])) # Shorthand for PadLastDim
        # self.init_transform = get_pad_last_dim(diff)
        # self.init_transform = PadLastDim(transformed_column_height)
        # self.init_transform = nn.Identity()
        # self.init_transform = nn.Upsample(size=(transformed_column_height), mode='linear')
        # self.init_transform = nn.ConvTranspose1d(n_channels, n_channels, kernel_size=2, stride=2, padding=7)

        self.inconv = (DoubleConv(n_channels, n_features, bn1=bn1, bn2=bn2, activation=activation))

        downs = [Down(n_features*2**i, n_features*2**(i+1), bn1=bn1, bn2=bn2, activation=activation) for i in range(n_levels-1)]
        downs = downs + [Down(n_features*2**(n_levels-1), n_features*2**(n_levels) // factor, bn1=bn1, bn2=bn2, activation=activation)]
        ups = [Up(n_features*2**(i+1) // factor, n_features*2**i, linear, bn1=bn1, bn2=bn2, activation=activation) for i in range(1,n_levels)]
        ups = [Up(n_features*2, n_features, linear, bn1=bn1, bn2=bn2, activation=activation)] + ups
        self.downs = nn.ModuleList(downs)
        self.ups = nn.ModuleList(ups)

        self.outconv = (OutConv(n_features, n_classes, activation))
        self.fc = nn.Linear(n_classes*transformed_column_height, output_channels_total)

    def forward(self, x):
        x = self.init_transform(x)
        x = self.inconv(x)

        # Downwards propagation with saving the outputs of each layer
        xdowns = []
        for down in self.downs:
            xdowns.append(x)
            x = down(x)

        # Upwards propagation with concatanation
        for up,xdown in list(zip(self.ups, xdowns))[::-1]:
            x = up(x, xdown)

        x = self.outconv(x)
        x = torch.flatten(x, 1)
        result = self.fc(x)
        return result

    def use_checkpointing(self):
        self.inconv = torch.utils.checkpoint(self.inconv)
        for i in range(len(self.ups)):
            self.ups[i] = torch.utils.checkpoint(self.ups[i])
            self.downs[i] = torch.utils.checkpoint(self.downs[i])
        self.outconv = torch.utils.checkpoint(self.outconv)


#-------------------------------------------------------------------------------------#
####################################### ResDNN #######################################
#-------------------------------------------------------------------------------------#


class ResDNNBlock(nn.Module):
    def __init__(self, n_layers, n_neurons, bn, activation):
        super(ResDNNBlock, self).__init__()
        self.fcs = [nn.Linear(n_neurons, n_neurons) for _ in range(n_layers)]
        self.bns = [nn.BatchNorm1d(n_neurons) if bn else nn.Identity() for _ in range(n_layers)]
        # activation = nn.ReLU()
        # print(activation)
        self.activations = [activation for _ in range(n_layers)]
        # self.activations = [nn.ReLU() for _ in range(n_layers)]

        self.block = nn.Sequential(*[element for layer in zip(self.fcs, self.bns, self.activations) for element in layer])

    def forward(self, x_in):
        x = self.block(x_in)
        return x + x_in

class ResDNN(nn.Module):
    def __init__(self, in_size, out_size, n_neurons, n_levels, n_layers_per_block, bn, activation=nn.ReLU()):
        super(ResDNN, self).__init__()
        self.activation = activation
        self.fc1 = nn.Linear(in_size, n_neurons)
        self.fc2 = nn.Linear(n_neurons, out_size)
        self.resdnnblocks = nn.ModuleList([ResDNNBlock(n_layers_per_block, n_neurons, bn, activation) for _ in range(n_levels)])

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.activation(self.fc1(x))
        for block in self.resdnnblocks:
            x = block(x)

        x = self.fc2(x)
        return x
