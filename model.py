import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_encoder

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 5, 7), 'kernel size must be 3 or 7'
        padding = kernel_size//2

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BottleNeck(nn.Module):
    def __init__(self, inplanes, width, planes, groups, stride=2):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.res = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,
                               stride=stride)

    def forward(self, x):

        res = self.res(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = out + res
        out = self.relu(out)

        return out

class CBAM(nn.Module):
    def __init__(self, channel, kernel_size=7) -> None:
        super(CBAM, self).__init__()
        
        # self.conv = BottleNeck(channel,channel//2,channel,32,1)
        self.ca1 = ChannelAttention(channel)
        self.sa1 = SpatialAttention(kernel_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x1 = self.conv(x)
        x1 = self.ca1(x) * x
        x1 = self.sa1(x1) * x1
        out = self.relu(x1+x)
        return out


class ReconstructiveSubNetwork_CBAM(nn.Module):
    def __init__(self,in_channels=3, out_channels=3, base_width=128):
        super(ReconstructiveSubNetwork_CBAM, self).__init__()
        self.encoder = get_encoder(
            "se_resnext50_32x4d",
            in_channels=in_channels,
            depth=4,
            weights="imagenet",
        )
        self.down1 = BottleNeck(1024, 1024, 1024, 32, 2)
        # self.down1_1 = BottleNeck(1024, 1024, 2048, 32, 1)
        self.cbam1 = CBAM(1024,5)
        self.down2 = BottleNeck(1024, 1024, 1024, 32, 2)
        # self.down2_1 = BottleNeck(2048, 1024, 2048, 32, 1)
        self.cbam2 = CBAM(1024,3)

        self.unsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.unsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.decoder1 = BottleNeck(2048, 1024, 1024, 32, 1)
        self.cbam3 = CBAM(1024,7)

        self.decoder2 = DecoderReconstructive(base_width, out_channels=out_channels)

    def forward(self, x):
        *_, x = self.encoder(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)

        x1 = self.cbam1(x1)
        x2 = self.cbam2(x2)
        x1 = self.unsample1(x1)
        x2 = self.unsample2(x2)
        x3 = torch.cat((x1,x2),dim=1)
        x3 = self.decoder1(x3)
        x3 = self.cbam3(x3)

        output = self.decoder2(x3)
        return output

class DiscriminativeSubNetwork2(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(DiscriminativeSubNetwork2, self).__init__()
        self.model = smp.Unet(
            in_channels=in_channels,
            encoder_name="timm-mobilenetv3_large_100",
            encoder_weights="imagenet",
            classes=out_channels,
        )
    def forward(self, x):
        return self.model(x)

class DiscriminativeSubNetwork(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(DiscriminativeSubNetwork, self).__init__()
        self.model = smp.Unet(
            in_channels=in_channels,
            encoder_name="se_resnext50_32x4d",
            encoder_weights="imagenet",
            classes=out_channels,
        )
    def forward(self, x):
        return self.model(x)

class DecoderReconstructive(nn.Module):
    def __init__(self, base_width, out_channels=1):
        super(DecoderReconstructive, self).__init__()
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 BottleNeck(inplanes=base_width * 8, width=base_width * 4, planes=base_width * 8, groups=32, stride=1),
                                #  BottleNeck(inplanes=base_width * 8, width=base_width * 4, planes=base_width * 8, groups=32, stride=1),
                                 BottleNeck(inplanes=base_width * 8, width=base_width * 4, planes=base_width * 4, groups=32, stride=1),
                                )
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 BottleNeck(inplanes=base_width * 4, width=base_width * 2, planes=base_width * 4, groups=32, stride=1),
                                #  BottleNeck(inplanes=base_width * 4, width=base_width * 2, planes=base_width * 4, groups=32, stride=1),
                                 BottleNeck(inplanes=base_width * 4, width=base_width * 2, planes=base_width * 2, groups=32, stride=1),
                                )
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 BottleNeck(inplanes=base_width * 2, width=base_width * 1, planes=base_width * 2, groups=16, stride=1),
                                #  BottleNeck(inplanes=base_width * 2, width=base_width * 1, planes=base_width * 2, groups=32, stride=1),
                                 BottleNeck(inplanes=base_width * 2, width=base_width * 1, planes=base_width * 1, groups=16, stride=1),
                                )
        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 BottleNeck(inplanes=base_width * 1, width=base_width * 1, planes=base_width * 1, groups=16, stride=1),
                                #  BottleNeck(inplanes=base_width * 1, width=base_width * 1, planes=base_width * 1, groups=32, stride=1),
                                 BottleNeck(inplanes=base_width * 1, width=base_width * 1, planes=base_width * 1, groups=16, stride=1),
                                )
        self.fin_out = nn.Sequential(nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1))
    def forward(self, x):
        x = self.up1(x)

        x = self.up2(x)

        x = self.up3(x)

        x = self.up4(x)

        out = self.fin_out(x)
        return out

# class DecoderReconstructive(nn.Module):
#     def __init__(self, base_width, out_channels=1):
#         super(DecoderReconstructive, self).__init__()

#         self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#                                  nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
#                                  nn.BatchNorm2d(base_width * 8),
#                                  nn.ReLU(inplace=True))
#         self.db1 = nn.Sequential(
#             nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
#             nn.BatchNorm2d(base_width*8),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(base_width * 8, base_width * 4, kernel_size=3, padding=1),
#             nn.BatchNorm2d(base_width * 4),
#             nn.ReLU(inplace=True)
#         )

#         self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#                                  nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
#                                  nn.BatchNorm2d(base_width * 4),
#                                  nn.ReLU(inplace=True))
#         self.db2 = nn.Sequential(
#             nn.Conv2d(base_width*4, base_width*4, kernel_size=3, padding=1),
#             nn.BatchNorm2d(base_width*4),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
#             nn.BatchNorm2d(base_width * 2),
#             nn.ReLU(inplace=True)
#         )

#         self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#                                  nn.Conv2d(base_width * 2, base_width*2, kernel_size=3, padding=1),
#                                  nn.BatchNorm2d(base_width*2),
#                                  nn.ReLU(inplace=True))
#         # cat with base*1
#         self.db3 = nn.Sequential(
#             nn.Conv2d(base_width*2, base_width*2, kernel_size=3, padding=1),
#             nn.BatchNorm2d(base_width*2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(base_width*2, base_width*1, kernel_size=3, padding=1),
#             nn.BatchNorm2d(base_width*1),
#             nn.ReLU(inplace=True)
#         )

#         self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#                                  nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
#                                  nn.BatchNorm2d(base_width),
#                                  nn.ReLU(inplace=True))
#         self.db4 = nn.Sequential(
#             nn.Conv2d(base_width*1, base_width, kernel_size=3, padding=1),
#             nn.BatchNorm2d(base_width),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
#             nn.BatchNorm2d(base_width),
#             nn.ReLU(inplace=True)
#         )

#         self.fin_out = nn.Sequential(nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1))
#         #self.fin_out = nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1)

#     def forward(self, b5):
#         up1 = self.up1(b5)
#         db1 = self.db1(up1)

#         up2 = self.up2(db1)
#         db2 = self.db2(up2)

#         up3 = self.up3(db2)
#         db3 = self.db3(up3)

#         up4 = self.up4(db3)
#         db4 = self.db4(up4)

#         out = self.fin_out(db4)
#         return out

if __name__=="__main__":
    from torchsummary import summary
    import time
    model = ReconstructiveSubNetwork_CBAM()
    model.cuda()
    summary(model,(3,256,256))
    z = torch.randn((4,3,256,256)).cuda()
    t1 = time.time()
    for i in range(200):
        y = model(z)
        del y
    t2 = time.time()
    print(t2-t1)