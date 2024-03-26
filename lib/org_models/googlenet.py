import torch
import torch.nn as nn


class Inception(nn.Module):
    def __init__(self, in_planes, kernel_1_x, kernel_3_in_cfg_1, kernel_3_x, kernel_5_in_x, kernel_5_x, pool_planes):
        super(Inception, self).__init__()
        
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, kernel_1_x, kernel_size=1),
            nn.BatchNorm2d(kernel_1_x),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, kernel_3_in_cfg_1, kernel_size=1),
            nn.BatchNorm2d(kernel_3_in_cfg_1),
            nn.ReLU(True),
            nn.Conv2d(kernel_3_in_cfg_1, kernel_3_x, kernel_size=3, padding=1),
            nn.BatchNorm2d(kernel_3_x),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, kernel_5_in_x[0], kernel_size=1),
            nn.BatchNorm2d(kernel_5_in_x[0]),
            nn.ReLU(True),
            nn.Conv2d(kernel_5_in_x[0], kernel_5_in_x[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(kernel_5_in_x[1]),
            nn.ReLU(True),
            nn.Conv2d(kernel_5_in_x[1], kernel_5_x, kernel_size=3, padding=1),
            nn.BatchNorm2d(kernel_5_x),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10, cfg=None):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        
        if cfg is None:
            cfg = [[96,  16, 32],
                   [128, 32, 96],
                   [96,  16, 48],
                   [112, 24, 64],
                   [128, 24, 64],
                   [144, 32, 64],
                   [160, 32, 128],
                   [160, 32, 128],
                   [192, 48, 128]]
            
        cfg = torch.tensor(cfg).view(9,3).tolist()
        self.cfg = cfg
        
        # self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        # self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)
        self.a3 = Inception(192,  64, cfg[0][0], 128, cfg[0][1:], 32, 32)
        self.b3 = Inception(256, 128, cfg[1][0], 192, cfg[1][1:], 96, 64)

        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)

        # self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        # self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        # self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        # self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        # self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a4 = Inception(480, 192, cfg[2][0], 208, cfg[2][1:],  48,  64)
        self.b4 = Inception(512, 160, cfg[3][0], 224, cfg[3][1:],  64,  64)
        self.c4 = Inception(512, 128, cfg[4][0], 256, cfg[4][1:],  64,  64)
        self.d4 = Inception(512, 112, cfg[5][0], 288, cfg[5][1:],  64,  64)
        self.e4 = Inception(528, 256, cfg[6][0], 320, cfg[6][1:], 128, 128)

        # self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        # self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.a5 = Inception(832, 256, cfg[7][0], 320, cfg[7][1:], 128, 128)
        self.b5 = Inception(832, 384, cfg[8][0], 384, cfg[8][1:], 128, 128)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.a3(x)
        x = self.b3(x)
        x = self.max_pool(x)
        x = self.a4(x)
        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        x = self.e4(x)
        x = self.max_pool(x)
        x = self.a5(x)
        x = self.b5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x