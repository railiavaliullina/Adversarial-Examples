import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from enum import Enum


class ModelType(Enum):
    BASELINE = 0
    IMPROVED = 1


class FirstConv(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        """
        :param cfg: cfg['model']['FirstConv'] part of config
        """
        self.in_channels = cfg['in_channels']
        self.out_channels = cfg['out_channels']
        self.kernel_size = cfg['kernel_size']
        self.stride = cfg['stride']
        self.padding = cfg['padding']
        self.bias = cfg['bias']

        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                              kernel_size=self.kernel_size,
                              stride=self.stride, padding=self.padding, bias=self.bias)
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU(inplace=True)

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class FirstConvC(nn.Module):  # for improved model
    def __init__(self, cfg):
        super().__init__()
        """
        :param cfg: cfg['model']['FirstConv'] part of config
        """
        self.in_channels = cfg['in_channels']
        self.out_channels = cfg['out_channels']
        self.kernel_size = cfg['kernel_size']
        self.stride = cfg['stride']
        self.padding = cfg['padding']
        self.bias = cfg['bias']

        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                               kernel_size=self.kernel_size,
                               stride=self.stride, padding=self.padding, bias=self.bias)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.out_channels_1 = self.out_channels * 2
        self.stride_1 = self.stride // 2
        self.conv2 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels_1,
                               kernel_size=self.kernel_size,
                               stride=self.stride_1, padding=self.padding, bias=self.bias)
        self.bn2 = nn.BatchNorm2d(self.out_channels_1)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels=self.out_channels_1, out_channels=self.out_channels_1,
                               kernel_size=self.kernel_size,
                               stride=self.stride_1, padding=self.padding, bias=self.bias)
        self.bn3 = nn.BatchNorm2d(self.out_channels_1)
        self.relu3 = nn.ReLU(inplace=True)

    def __call__(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        return x


class Bottleneck(nn.Module):
    def __init__(self, cfg, in_channels, out_channels, stride=1, is_downsampling=False):
        super(Bottleneck, self).__init__()
        """
        :param cfg: cfg['model']['LayersGroup']['BottleNeck'] part of config
        """
        self.kernel_size = cfg['kernel_size']
        self.padding = cfg['padding']
        self.bias = cfg['bias']

        out_channels_2 = 4 * out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size[0], bias=self.bias,
                               stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=self.kernel_size[1], padding=self.padding,
                               bias=self.bias)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels_2, kernel_size=self.kernel_size[2], bias=self.bias)
        self.bn3 = nn.BatchNorm2d(out_channels_2)
        self.relu = nn.ReLU(inplace=True)
        if is_downsampling:
            self.downsample = nn.Sequential(*[nn.Conv2d(in_channels, out_channels_2,
                                                        kernel_size=cfg['downsample']['kernel_size'], stride=stride,
                                                        bias=cfg['downsample']['bias']),
                                              nn.BatchNorm2d(out_channels_2)])
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        conv1 = self.relu(self.bn1(self.conv1(x)))
        conv2 = self.relu(self.bn2(self.conv2(conv1)))
        conv3 = self.bn3(self.conv3(conv2))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = conv3 + identity
        return self.relu(out)


class BottleneckImproved(nn.Module):  # for improved model
    def __init__(self, cfg, in_channels, out_channels, stride=1, is_downsampling=False):
        super(BottleneckImproved, self).__init__()
        """
        :param cfg: cfg['model']['LayersGroup']['BottleNeck'] part of config
        """
        self.kernel_size = cfg['kernel_size']
        self.padding = cfg['padding']
        self.bias = cfg['bias']

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size[0], bias=self.bias)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=self.kernel_size[1], padding=self.padding,
                               bias=self.bias, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_channels)

        out_channels_2 = 4 * out_channels
        self.conv3 = nn.Conv2d(out_channels, out_channels_2, kernel_size=self.kernel_size[2], bias=self.bias)
        self.bn3 = nn.BatchNorm2d(out_channels_2)
        self.bn3.running_var *= 0
        self.relu = nn.ReLU(inplace=True)

        if is_downsampling:
            self.downsample = nn.Sequential(*[nn.AvgPool2d(kernel_size=cfg['downsample']['avg_pool_kernel_size'],
                                                           stride=cfg['downsample']['avg_pool_stride']),
                                              nn.Conv2d(in_channels, out_channels_2,
                                                        kernel_size=cfg['downsample']['kernel_size'],
                                                        bias=cfg['downsample']['bias']),
                                              nn.BatchNorm2d(out_channels_2)])
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        conv1 = self.bn1(self.conv1(x))
        conv2 = self.bn2(self.conv2(conv1))
        conv3 = self.bn3(self.conv3(conv2))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = conv3 + identity
        return self.relu(out)


class LayersGroup(nn.Module):
    def __init__(self, cfg, name, model_type):
        super(LayersGroup, self).__init__()
        """
        :param cfg: cfg['model']['LayersGroup'] part of config
        :param name: name of current layer group
        :param model_type: type of using model (baseline or improved)
        """
        BottleneckClass = Bottleneck if model_type == 'BASELINE' or name == 'layer1' else BottleneckImproved
        self.in_channels = cfg[name]['in_channels']
        self.out_channels = cfg[name]['out_channels']
        self.nb_layers = cfg[name]['nb_layers']
        self.stride = cfg[name]['stride']

        self.out_channels_2 = 4 * self.out_channels
        self.layers_group = [
            BottleneckClass(cfg['BottleNeck'], self.in_channels, self.out_channels, stride=self.stride,
                            is_downsampling=True)]
        for _ in range(1, self.nb_layers):
            self.layers_group.append(BottleneckClass(cfg['BottleNeck'], self.out_channels_2, self.out_channels))
        self.layers_group = nn.Sequential(*self.layers_group)


class ResNet50(nn.Module):
    def __init__(self, cfg, model_type):
        super(ResNet50, self).__init__()
        """
        Collects all parts of ResNet50 model
        :param cfg: cfg['model'] part of config
        :param model_type: type of using model (baseline or improved)
        """
        self.conv1 = FirstConv(cfg['FirstConv']) if model_type == 'BASELINE' else FirstConvC(cfg['FirstConv'])
        self.maxpool = nn.MaxPool2d(kernel_size=cfg['MaxPool']['kernel_size'], stride=cfg['MaxPool']['stride'],
                                    padding=cfg['MaxPool']['padding'])

        self.layer1 = LayersGroup(cfg['LayersGroup'], name='layer1', model_type=model_type).layers_group
        self.layer2 = LayersGroup(cfg['LayersGroup'], name='layer2', model_type=model_type).layers_group
        self.layer3 = LayersGroup(cfg['LayersGroup'], name='layer3', model_type=model_type).layers_group
        self.layer4 = LayersGroup(cfg['LayersGroup'], name='layer4', model_type=model_type).layers_group

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=cfg['AvgPool']['output_size'])
        self.fc = nn.Linear(in_features=cfg['Linear']['in_features'], out_features=cfg['Linear']['out_features'],
                            bias=cfg['Linear']['bias'])

    def forward(self, x):
        conv1 = self.conv1(x)
        maxpool = self.maxpool(conv1)
        layer1 = self.layer1(maxpool)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        avgpool = self.avgpool(layer4)
        fc = self.fc(torch.flatten(avgpool, 1))
        return fc


def get_model(cfg, model_type):
    """
    Gets ResNet-50 model
    :param cfg: cfg['model'] part of config
    :param model_type: type of using model (baseline or improved)
    :return: ResNet-50 model
    """
    model = ResNet50(cfg, model_type)

    nb_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters number: {nb_trainable_params}')

    if cfg['pretrained']['load_pretrained']:
        print(f'Loading pretrained weights to initialize model...')
        state_dict = load_state_dict_from_url(cfg['pretrained']['url'], progress=cfg['pretrained']['progress'])
        model.load_state_dict(state_dict)
    else:
        print(f'Initializing weights with xavier uniform...')
        model_parameters = model.parameters()
        for i, param in enumerate(model_parameters):
            if len(param.size()) == 4:
                torch.nn.init.xavier_uniform_(param)
    return model
