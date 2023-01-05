
import torch
import torch.nn as nn


def conv3(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetFeature(nn.Module):

    def __init__(self, config):

        super(ResNetFeature, self).__init__()

        self.layer_config_dict = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3]
        }
        self.inplanes = 16
        self.config = config
        self.layers = self.layer_config_dict[config['NUM_LAYER']]
        if config['NUM_LAYER'] == 18 or config['NUM_LAYER'] == 34:
            block = BasicBlock
            outsize = 32*16
        elif config['NUM_LAYER'] == 50 or config['NUM_LAYER'] == 101 or config['nun_layers'] == 152:
            block = Bottleneck
            outsize = 128*16
        else:
            raise NotImplementedError("num layers '{}' is not in layer config".format(config['num_layers']))

        self.initial_layer = nn.Sequential(
            nn.Conv1d(1, 16, 7, 2, 3, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(3, 2, 1))

        self.layer1 = self._make_layer(block, 16, self.layers[0], stride=1, first=True)
        self.layer2 = self._make_layer(block, 16, self.layers[1], stride=2)
        self.layer3 = self._make_layer(block, 32, self.layers[2], stride=2)
        self.layer4 = self._make_layer(block, 32, self.layers[3], stride=2)
        self.maxpool = nn.MaxPool1d(3, 2, 1)

        self.dropout = nn.Dropout(p=config['DROPOUT_RATE'])
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # outsize
        # 18, 34: (b, 32, 16)
        # 50, 101: (b, 128, 16)
        self.decoder = nn.Linear(outsize, config["FEATURE"])

    def _make_layer(self, block, planes, blocks, stride=1, first=False):
        print(block, planes, blocks, stride, first)
        downsample = None
        if (stride != 1 and first is False) or self.inplanes != planes * block.expansion:
            print("... DOWN", self.inplanes, planes * block.expansion, stride)
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion)
            )

        layers = []
        print("... 0", self.inplanes, planes, stride)
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            print("...", _, self.inplanes, planes)
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # shape must be (batch, channel, data)
        if len(x.shape) == 2: 
            x = x.unsqueeze(1)

        f = self.initial_layer(x)
        f = self.layer1(f)
        f = self.layer2(f)
        f = self.maxpool(f)
        f = self.layer3(f)
        f = self.layer4(f)
        f = self.dropout(f)
        f = f.flatten(1, -1)
        out = self.decoder(f)
        return out