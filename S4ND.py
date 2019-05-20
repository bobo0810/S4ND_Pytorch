import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm3d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=(3, 3, 3), stride=1, padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        # 减少输入的feature map数量。 一般为growth rate*4
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm3d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_planes, inter_planes, kernel_size=(3, 3, 3), stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(inter_planes)
        self.conv2 = nn.Conv3d(inter_planes, out_planes, kernel_size=(3, 3, 3), stride=1, padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm3d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=(3, 3, 3), stride=1, padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return out

class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)
    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class S4ND(nn.Module):
    '''
    growth_rate 增长率  block中每层输出的特征图channel
    block内 每个操作输入不断增加，输出为growth_rate不变
    '''
    def __init__(self):
        super(S4ND, self).__init__()
        # Translation layer 层通道数减少比例，默认减半
        reduction = 0.5
        # 使用bottleneck layer
        bottleneck = True
        # 最后一个conv卷积层1x1x1的输出channel
        end_out_planes=1
        # 不使用dropout
        dropRate = 0.0
        # 5 个Dense Block的Growth rate
        growth_rate=[16,16,16,32,64]
        # 5 个Dense Block内的卷积层数量
        block_conv_num = [6,6,6,6,6]
        # 整体网络初始化的输入channel
        init_in_planes=1
        if bottleneck == True:
            block = BottleneckBlock
        else:
            block = BasicBlock




        ######第一个Dense Block#####
        self.block1 = DenseBlock(block_conv_num[0], init_in_planes, growth_rate[0], block, dropRate)
        # 计算 Translation layer的输入，为 初始数据的channel + 本次Block输出的channel
        in_planes = int(init_in_planes + block_conv_num[0] * growth_rate[0])
        # Translation layer : block之间增加了1*1的卷积操作，默认输出channel减半
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes * reduction)), dropRate=dropRate)
        #  Translation layer  输出的channel
        in_planes = int(math.floor(in_planes*reduction))

        ######第二个Dense Block#####
        self.block2 = DenseBlock(block_conv_num[1], in_planes, growth_rate[1], block, dropRate)
        # 计算 Translation layer的输入,为 上一个block的输出channel +  本次Block输出的channel
        in_planes = int(in_planes + block_conv_num[1] * growth_rate[1])
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))

        ######第三个Dense Block#####
        self.block3 = DenseBlock(block_conv_num[2], in_planes, growth_rate[2], block, dropRate)
        # 计算 Translation layer的输入,为 上一个block的输出channel +  本次Block输出的channel
        in_planes = int(in_planes + block_conv_num[2] * growth_rate[2])
        self.trans3 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))

        ######第四个Dense Block#####
        self.block4 = DenseBlock(block_conv_num[3], in_planes, growth_rate[3], block, dropRate)
        # 计算 Translation layer的输入,为 上一个block的输出channel +  本次Block输出的channel
        in_planes = int(in_planes + block_conv_num[3] * growth_rate[3])
        self.trans4 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))

        ######第五个Dense Block#####
        self.block5 = DenseBlock(block_conv_num[4], in_planes, growth_rate[4], block, dropRate)
        # 计算 Block的输出,为 上一个block的输出channel（本次block的输入） +  本次Block输出的channel
        in_planes = int(in_planes + block_conv_num[4] * growth_rate[4])

        ######Conv 1x1x1  不改变大小##### ok
        self.end_conv = torch.nn.Conv3d(in_channels=in_planes, out_channels=end_out_planes, kernel_size=(1, 1, 1))

        # 最大池化层(根据论文中网络图，第一个池化操作后 特征图缩小明显，故第一个maxpool卷积核比其余maxpool大)
        self.maxpool_begin = torch.nn.MaxPool3d(kernel_size=(1, 4, 4))
        self.maxpool = torch.nn.MaxPool3d(kernel_size=(1, 2, 2))
    def forward(self, x):

        out = self.maxpool_begin(self.trans1(self.block1(x)))
        out = self.maxpool(self.trans2(self.block2(out)))
        out = self.maxpool(self.trans3(self.block3(out)))
        out = self.maxpool(self.trans4(self.block4(out)))
        out = self.block5(out)
        out = self.end_conv(out)
        # sigmod

        return out


if __name__ == '__main__':
    model=S4ND()
    input = torch.randn(2, 1, 8,512, 512)  # 2 batch size    1  输入通道     26，40,40为一个通道的样本
    output=model(input)
    print(output.size())