import torch.nn as nn
import torch
import torch.nn.functional as F
from hdconv import HDConv 

class SAMDConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilations=(1, 2, 3, 5),  stride=1,
                 padding=0,  bias=False):
        super(SAMDConv, self).__init__()
        # dconv
        self.dconv_1 = nn.Conv2d(in_channels,
                                 out_channels//4,
                                 kernel_size,
                                 stride=stride,
                                 padding=int(padding + (dilations[0] - 1) * (kernel_size - 1)//2),
                                 dilation=dilations[0],
                                 bias=bias)
        self.dconv_2 = nn.Conv2d(in_channels,
                                 out_channels//4,
                                 kernel_size,
                                 stride=stride,
                                 padding=int(padding + (dilations[1] - 1) * (kernel_size - 1)//2),
                                 dilation=dilations[1],
                                 bias=bias)
        self.dconv_3 = nn.Conv2d(in_channels,
                                 out_channels//4,
                                 kernel_size,
                                 stride=stride,
                                 padding=int(padding + (dilations[2] - 1) * (kernel_size - 1)//2),
                                 dilation=dilations[2],
                                 bias=bias)
        self.dconv_4 = nn.Conv2d(in_channels,
                                 out_channels//4,
                                 kernel_size,
                                 stride=stride,
                                 padding=int(padding + (dilations[3] - 1) * (kernel_size - 1)//2),
                                 dilation=dilations[3],
                                 bias=bias)
        # attention
        self.spatial_attention = nn.Sequential(
            HDConv(in_channels, 16, kernel_size, stride=stride, padding=padding, bias=bias,dilation=dilations),
            nn.SyncBatchNorm(16),
            nn.Sigmoid()
        )

    def forward(self, x):
        # dilated conv
        dout_1 = self.dconv_1(x)
        dout_2 = self.dconv_2(x)
        dout_3 = self.dconv_3(x)
        dout_4 = self.dconv_4(x)
        # # attention
        spatial_prob = self.spatial_attention(x)
        # print(spatial_prob)
        # softmax
        spatial_prob=F.softmax(spatial_prob,dim=1)
        # print(spatial_prob)
        probs = torch.split(spatial_prob, 1, dim=1)

        # feature fusion
        output = dout_1 * probs[0] + dout_2 * probs[1] + dout_3 * probs[2] + dout_4 * probs[3]
        for i in range(1, 4):
            n = 4 * i
            output = torch.cat((output, dout_1 * probs[n] +
                                dout_2 * probs[n + 1] + dout_3 * probs[n + 2] + dout_4 * probs[n + 3]), dim=1)
        return output
