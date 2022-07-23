import torch
import cupy_module.adacof as adacof
import sys
from torch.nn import functional as F
from utility import CharbonnierFunc, moduleNormalize



def make_model(args):
    return MESConv(args).cuda()

def conv5x5(in_channels, out_channels, stride=1, 
            padding=2, bias=True, groups=1):    
    return torch.nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=5,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)


class KernelEstimation(torch.nn.Module):
    def __init__(self, kernel_size):
        super(KernelEstimation, self).__init__()
        self.kernel_size = kernel_size

        def Basic(input_channel, output_channel):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )

        def Upsample(channel):
            return torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )

        def Subnet_offset(ks):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=ks, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=ks, out_channels=ks, kernel_size=3, stride=1, padding=1)
            )

        def Subnet_weight(ks):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=ks, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=ks, out_channels=ks, kernel_size=3, stride=1, padding=1),
                torch.nn.Softmax(dim=1)
            )

        def Subnet_occlusion():
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),
                torch.nn.Sigmoid()
            )

        def Subnet_map1():
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),
                torch.nn.Sigmoid()
            )

        def Subnet_map2():
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),
                torch.nn.Sigmoid()
            )

        self.moduleConv1 = Basic(6, 32)
        self.modulePool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = Basic(32, 64)
        self.modulePool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv3 = Basic(64, 128)
        self.modulePool3 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = Basic(128, 256)
        self.modulePool4 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv5 = Basic(256, 512)
        self.modulePool5 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleDeconv5 = Basic(512, 512)
        self.moduleUpsample5 = Upsample(512)

        self.moduleDeconv4 = Basic(512, 256)
        self.moduleUpsample4 = Upsample(256)

        self.moduleDeconv3 = Basic(256, 128)
        self.moduleUpsample3 = Upsample(128)

        self.moduleDeconv2 = Basic(128, 64)
        self.moduleUpsample2 = Upsample(64)

        self.moduleWeight11 = Subnet_weight(self.kernel_size ** 2)
        self.moduleAlpha11 = Subnet_offset(self.kernel_size ** 2)
        self.moduleBeta11 = Subnet_offset(self.kernel_size ** 2)
        self.moduleWeight21 = Subnet_weight(self.kernel_size ** 2)
        self.moduleAlpha21 = Subnet_offset(self.kernel_size ** 2)
        self.moduleBeta21 = Subnet_offset(self.kernel_size ** 2)
        self.moduleWeight12 = Subnet_weight(self.kernel_size ** 2)
        self.moduleAlpha12 = Subnet_offset(self.kernel_size ** 2)
        self.moduleBeta12 = Subnet_offset(self.kernel_size ** 2)
        self.moduleWeight22 = Subnet_weight(self.kernel_size ** 2)
        self.moduleAlpha22 = Subnet_offset(self.kernel_size ** 2)
        self.moduleBeta22 = Subnet_offset(self.kernel_size ** 2)
        self.moduleWeight13 = Subnet_weight(self.kernel_size ** 2)
        self.moduleAlpha13 = Subnet_offset(self.kernel_size ** 2)
        self.moduleBeta13 = Subnet_offset(self.kernel_size ** 2)
        self.moduleWeight23 = Subnet_weight(self.kernel_size ** 2)
        self.moduleAlpha23 = Subnet_offset(self.kernel_size ** 2)
        self.moduleBeta23 = Subnet_offset(self.kernel_size ** 2)
        self.moduleOcclusion1 = Subnet_occlusion()
        self.moduleOcclusion2 = Subnet_occlusion()
        self.moduleOcclusion3 = Subnet_occlusion()
        self.modulemap1 = Subnet_map1()
        self.modulemap2 = Subnet_map2()

    def forward(self, rfield0, rfield2):
        tensorJoin = torch.cat([rfield0, rfield2], 1)

        tensorConv1 = self.moduleConv1(tensorJoin)
        tensorPool1 = self.modulePool1(tensorConv1)

        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)


        tensorConv4 = self.moduleConv4(tensorPool3)
        tensorPool4 = self.modulePool4(tensorConv4)

        tensorConv5 = self.moduleConv5(tensorPool4)
        tensorPool5 = self.modulePool5(tensorConv5)

        tensorDeconv5 = self.moduleDeconv5(tensorPool5)
        tensorUpsample5 = self.moduleUpsample5(tensorDeconv5)

        tensorCombine = tensorUpsample5 + tensorConv5

        tensorDeconv4 = self.moduleDeconv4(tensorCombine)
        tensorUpsample4 = self.moduleUpsample4(tensorDeconv4)

        tensorCombine = tensorUpsample4 + tensorConv4

        tensorDeconv3 = self.moduleDeconv3(tensorCombine)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)

        tensorCombine = tensorUpsample3 + tensorConv3

        tensorDeconv2 = self.moduleDeconv2(tensorCombine)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)

        tensorCombine = tensorUpsample2 + tensorConv2

        Occlusion1 = self.moduleOcclusion1(tensorCombine)
        Occlusion2 = self.moduleOcclusion2(tensorCombine)
        Occlusion3 = self.moduleOcclusion3(tensorCombine)

        map1 = self.modulemap1(tensorCombine)
        map2 = self.modulemap2(tensorCombine)

        Weight11 = self.moduleWeight11(tensorCombine)
        Weight12 = self.moduleWeight12(tensorCombine)
        Weight13 = self.moduleWeight13(tensorCombine)

        Weight21 = self.moduleWeight21(tensorCombine)
        Weight22 = self.moduleWeight22(tensorCombine)
        Weight23 = self.moduleWeight23(tensorCombine)

        Alpha11 = self.moduleAlpha11(tensorCombine)
        Alpha12 = self.moduleAlpha12(tensorCombine)
        Alpha13 = self.moduleAlpha13(tensorCombine)

        Alpha21 = self.moduleAlpha21(tensorCombine)
        Alpha22 = self.moduleAlpha22(tensorCombine)
        Alpha23 = self.moduleAlpha23(tensorCombine)

        Beta11 = self.moduleBeta11(tensorCombine)
        Beta12 = self.moduleBeta12(tensorCombine)
        Beta13 = self.moduleBeta13(tensorCombine)

        Beta21 = self.moduleBeta21(tensorCombine)
        Beta22 = self.moduleBeta22(tensorCombine)
        Beta23 = self.moduleBeta23(tensorCombine)

        return Weight11, Alpha11, Beta11, Weight21, Alpha21, Beta21, Weight12, Alpha12, Beta12, Weight22, Alpha22, Beta22, Weight13, Alpha13, Beta13, Weight23, Alpha23, Beta23, Occlusion1, Occlusion2, Occlusion3, map1, map2


class MESConv(torch.nn.Module):
    def __init__(self, args):
        super(MESConv, self).__init__()
        self.args = args
        self.kernel_size = args.kernel_size
        self.kernel_pad1 = int(((args.kernel_size - 1) * args.dilation1) / 2.0)
        self.kernel_pad2 = int(((args.kernel_size - 1) * args.dilation2) / 2.0)
        self.kernel_pad3 = int(((args.kernel_size - 1) * args.dilation3) / 2.0)
        self.dilation1 = args.dilation1
        self.dilation2 = args.dilation2
        self.dilation3 = args.dilation3

        self.get_kernel = KernelEstimation(self.kernel_size)

        self.modulePad1 = torch.nn.ReplicationPad2d([self.kernel_pad1, self.kernel_pad1, self.kernel_pad1, self.kernel_pad1])
        self.modulePad2 = torch.nn.ReplicationPad2d([self.kernel_pad2, self.kernel_pad2, self.kernel_pad2, self.kernel_pad2])
        self.modulePad3 = torch.nn.ReplicationPad2d([self.kernel_pad3, self.kernel_pad3, self.kernel_pad3, self.kernel_pad3])

        self.moduleAdaCoF = adacof.FunctionAdaCoF.apply

        # self.Conv5 = torch.nn.Sequential(conv5x5(3, 32), torch.nn.ReLU(), conv5x5(32,3), torch.nn.Tanh())


    def forward(self, frame0, frame2):
        h0 = int(list(frame0.size())[2])
        w0 = int(list(frame0.size())[3])
        h2 = int(list(frame2.size())[2])
        w2 = int(list(frame2.size())[3])
        if h0 != h2 or w0 != w2:
            sys.exit('Frame sizes do not match')

        h_padded = False
        w_padded = False
        if h0 % 32 != 0:
            pad_h = 32 - (h0 % 32)
            frame0 = F.pad(frame0, (0, 0, 0, pad_h), mode='reflect')
            frame2 = F.pad(frame2, (0, 0, 0, pad_h), mode='reflect')
            h_padded = True

        if w0 % 32 != 0:
            pad_w = 32 - (w0 % 32)
            frame0 = F.pad(frame0, (0, pad_w, 0, 0), mode='reflect')
            frame2 = F.pad(frame2, (0, pad_w, 0, 0), mode='reflect')
            w_padded = True


        Weight11, Alpha11, Beta11, Weight21, Alpha21, Beta21, Weight12, Alpha12, Beta12, Weight22, Alpha22, Beta22, Weight13, Alpha13, Beta13, Weight23, Alpha23, Beta23, Occlusion1, Occlusion2, Occlusion3, map1, map2 = self.get_kernel(moduleNormalize(frame0), moduleNormalize(frame2))

        tensorAdaCoFp1a = self.moduleAdaCoF(self.modulePad1(frame0), Weight11, Alpha11, Beta11, self.dilation1)
        tensorAdaCoFp1b = self.moduleAdaCoF(self.modulePad1(frame2), Weight21, Alpha21, Beta21, self.dilation1)

        tensorAdaCoFp2a = self.moduleAdaCoF(self.modulePad2(frame0), Weight12, Alpha12, Beta12, self.dilation2)
        tensorAdaCoFp2b = self.moduleAdaCoF(self.modulePad2(frame2), Weight22, Alpha22, Beta22, self.dilation2)

        tensorAdaCoFp3a = self.moduleAdaCoF(self.modulePad3(frame0), Weight13, Alpha13, Beta13, self.dilation3)
        tensorAdaCoFp3b = self.moduleAdaCoF(self.modulePad3(frame2), Weight23, Alpha23, Beta23, self.dilation3)

        framep1 = Occlusion1 * tensorAdaCoFp1a + (1 - Occlusion1) * tensorAdaCoFp1b
        framep2 = Occlusion2 * tensorAdaCoFp2a + (1 - Occlusion2) * tensorAdaCoFp2b
        framep3 = Occlusion3 * tensorAdaCoFp3a + (1 - Occlusion3) * tensorAdaCoFp3b

        frame1 = map1 * framep1 + map2 * framep2 + (1 - map1 - map2) * framep3    

        if h_padded:
            frame1 = frame1[:, :, 0:h0, :]
        if w_padded:
            frame1 = frame1[:, :, :, 0:w0]


        return frame1
