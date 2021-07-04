import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, resblock=True):
        super(BasicBlock, self).__init__()
        self.resblock = resblock
        self.conv = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.mp = nn.MaxPool2d(2, 2)
        self.bn = nn.BatchNorm2d(planes)

        if resblock:
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, planes, kernel_size=1,
                                   stride=2, padding=0, bias=False)

    def forward(self, x):
        out = F.relu(self.bn(self.mp(self.conv(x))))
        print('Basic {}'.format(out.shape))

        # Condition for shortcut
        if self.resblock:
            print('Basic r input {}'.format(x.shape))
            rout = F.relu(self.bn1(self.conv1(x)))
            print('Basic r{}'.format(rout.shape))
            rout = F.relu(self.bn2(self.conv2(rout)))
            print('Basic r{}'.format(rout.shape))
            rout = self.conv3(rout)

            out += rout

        return out


class MyResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(MyResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 128, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 256, num_blocks[1], stride=1, resblock=False)
        self.layer3 = self._make_layer(block, 512, num_blocks[2], stride=1)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, resblock=True):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, resblock))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        print('Prep Layer')
        out = F.relu(self.bn1(self.conv1(x)))
        print(out.shape)
        print("---------------Layer 1------------------")
        out = self.layer1(out)
        print("---------------Layer 2------------------")
        out = self.layer2(out)
        print("---------------Layer 3------------------")
        out = self.layer3(out)
        print("---------------------------------")
        out = F.max_pool2d(out, 4, 1)
        print("after MP {}".format(out.shape))

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def customResnet():
    return MyResNet(BasicBlock, [1,1,1])