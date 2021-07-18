# ERFNET full network definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class DownsamplerBlock1(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        # print("4444444444444444444444")
        self.conv1 = nn.Conv2d(ninput, noutput, (3, 1), stride=1, padding=(1 , 0), bias=True, dilation=(1, 1))
        self.conv2 = nn.Conv2d(noutput, noutput, (1, 3), stride=1, padding=(0, 1), bias=True, dilation=(1, 1))
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output1 = self.conv1(input)
        output2 = self.conv2(output1)
        # output = torch.cat([output1, output2], 1)
        output = self.pool(output2)
        output = self.bn(output)
        return F.relu(output)


class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        # print("4444444444444444444444")
        self.conv = nn.Conv2d(ninput, noutput - ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)

class dialation_operation(nn.Module):
    def __init__(self, chann, dilated):
        super().__init__()

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True,
                                   dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * dilated), bias=False,
                                   dilation=(1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

    def forward(self, input):
        output = self.conv3x1_2(input)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        # if (self.dropout.p != 0):
        #     output = self.dropout(output)
        return F.relu(output)
        # return F.relu(output + input)


class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True,
                                   dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * dilated), bias=False,
                                   dilation=(1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return F.relu(output + input)  # +input = identity (residual connection)
class conv_bottleneck(nn.Module):
    def __init__(self, incannel,out, dropprob, dilated):
        super().__init__()

        self.conv3 = nn.Conv2d(incannel, out, (3, 3), stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(out, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(out, out, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True,
                                   dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(out, out, (1, 3), stride=1, padding=(0, 1 * dilated), bias=False,
                                   dilation=(1, dilated))

        self.bn2 = nn.BatchNorm2d(out, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        output = self.conv3(input)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return F.relu(output)


class Encoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.initial_block = DownsamplerBlock1(3, 16)
        # self.layers0 = nn.ModuleList()
        # for x in range(0, 2):
        #     self.layers0.append(non_bottleneck_1d(16, 0.1, 1))

        self.block1 = DownsamplerBlock(16, 64)
        self.layers1_1 = dialation_operation(64, 1)
        self.layers1_2 = dialation_operation(64, 2)
        # self.layers1 = nn.ModuleList()
        # for x in range(0, 5):  # 5 times
        #     self.layers1.append(non_bottleneck_1d(64, 0.1, 1))

        self.conv1=conv_bottleneck(128,64,0.1,1)
        self.block2=DownsamplerBlock1(64, 128)
        self.message_passing = nn.ModuleList()
        self.message_passing.add_module('up_down', nn.Conv2d(128, 128, (1, 9), padding=(0, 9 // 2), bias=False))
        self.message_passing.add_module('down_up', nn.Conv2d(128, 128, (1, 9), padding=(0, 9 // 2), bias=False))
        # self.message_passing.add_module('left_right',
        #                                 nn.Conv2d(128, 128, (9, 1), padding=(9 // 2, 0), bias=False))
        # self.message_passing.add_module('right_left',
        #                                 nn.Conv2d(128, 128, (9, 1), padding=(9// 2, 0), bias=False))


        self.layers2 = non_bottleneck_1d(128, 0.1, 2)
        #for x in range(0, 2):  # 1 times
        #    self.layers2.append(non_bottleneck_1d(128, 0.1, 1))
        #    self.layers2.append(non_bottleneck_1d(128, 0.1, 2))
        #    self.layers2.append(non_bottleneck_1d(128, 0.1, 1))
        #    self.layers2.append(non_bottleneck_1d(128, 0.1, 2))
        self.conv2 = conv_bottleneck(256,128, 0.1, 1)
        # only for encoder mode:
        #self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):
        output = self.initial_block(input)
        output1=self.block1(output)
        output2 = self.layers1_1(output1)
        output2 = self.layers1_2(output2)
        output3=torch.cat([output2, output1], 1)
        output4=self.conv1(output3)
        output5=self.block2(output4)
        output6= self.message_passing_forward(output5)
        output7= self.layers2(output6)
        #for layer in self.layers2:
        #    output7= layer(output6)
        output8=torch.cat([output7, output5], 1)
        output9=self.conv2(output8)
        
        return output9

    def message_passing_forward(self, x):
        Vertical = [True, True]
        Reverse = [False, True]
        for ms_conv, v, r in zip(self.message_passing, Vertical, Reverse):
            x = self.message_passing_once(x, ms_conv, v, r)
        return x

    def message_passing_once(self, x, conv, vertical=True, reverse=False):
        """
        Argument:
        ----------
        x: input tensor
        vertical: vertical message passing or horizontal
        reverse: False for up-down or left-right, True for down-up or right-left
        """
        nB, C, H, W = x.shape
        if vertical:
            slices = [x[:, :, i:(i + 1), :] for i in range(H)]
            dim = 2
        else:
            slices = [x[:, :, :, i:(i + 1)] for i in range(W)]
            dim = 3
        if reverse:
            slices = slices[::-1]

        out = [slices[0]]
        for i in range(1, len(slices)):
            out.append(slices[i] + F.relu(conv(out[i - 1])))
        if reverse:
            out = out[::-1]
        return torch.cat(out, dim=dim)

class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d(16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output


class Lane_exist(nn.Module):
    def __init__(self, num_output):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(nn.Conv2d(128, 32, (3, 3), stride=1, padding=(4, 4), bias=False, dilation=(4, 4)))
        self.layers.append(nn.BatchNorm2d(32, eps=1e-03))

        self.layers_final = nn.ModuleList()

        self.layers_final.append(nn.Dropout2d(0.1))
        self.layers_final.append(nn.Conv2d(32, 5, (1, 1), stride=1, padding=(0, 0), bias=True))

        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.linear1 = nn.Linear(3965, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, 4)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = F.relu(output)

        for layer in self.layers_final:
            output = layer(output)

        output = F.softmax(output, dim=1)
        output = self.maxpool(output)
        # print(output.shape)
        output = output.view(-1, 3965)
        output = self.linear1(output)
        output = F.relu(output)
        output = self.linear2(output)
        output = F.relu(output)
        output = self.linear3(output)
        output = torch.sigmoid(output)

        return output


class ERFNet(nn.Module):
    def __init__(self, num_classes, partial_bn=False, encoder=None):  # use encoder to pass pretrained encoder
        super().__init__()

        if (encoder == None):
            self.encoder = Encoder(num_classes)
        else:
            self.encoder = encoder
        self.decoder = Decoder(num_classes)
        self.lane_exist = Lane_exist(4)  # num_output
        self.input_mean = [103.939, 116.779, 123.68]  # [0, 0, 0]
        self.input_std = [1, 1, 1]
        self._enable_pbn = partial_bn

    def forward(self, input, only_encode=False):
        '''if only_encode:
            return self.encoder.forward(input, predict=True)
        else:'''
        output = self.encoder(input)  # predict=False by default
        return self.decoder.forward(output), self.lane_exist(output)

