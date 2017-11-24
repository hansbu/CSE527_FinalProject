import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        x = self.relu(x)
        return x


class StackEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StackEncoder, self).__init__()
        self.convr1 = ConvBnRelu(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=0)
        self.convr2 = ConvBnRelu(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=0)
        self.maxPool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

    def forward(self, x):
        x = self.convr1(x)
        x = self.convr2(x)
        x_trace = x             # save x to use in concat in the Decoder path
        out = self.maxPool(x)
        return out, x_trace


class StackDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_size, padding, drop_out=False):
        super(StackDecoder, self).__init__()

        self.drop_out = drop_out
        self.dropout_layer = nn.Dropout2d(p=0.5)

        ''' this is old version
        self.upSample = nn.Upsample(size=upsample_size, scale_factor=(2, 2), mode="bilinear")
        self.convr1 = ConvBnRelu(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=padding)
        # Crop + concat step between these 2
        self.convr2 = ConvBnRelu(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=padding)
        '''
        self.upSample = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.convr1 = ConvBnRelu(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=padding)
        self.convr2 = ConvBnRelu(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=padding)

    def _crop_concat(self, upsampled, bypass):
        """
         Crop y to the (h, w) of x and concat them.
         Used for the expansive path.
        Returns:
            The concatenated tensor
        """
        c = (bypass.size()[2] - upsampled.size()[2]) // 2
        bypass = F.pad(bypass, (-c, -c, -c, -c))

        return torch.cat((bypass, upsampled), 1)

    def forward(self, x, down_tensor):
        '''old version
        x = self.upSample(x)
        x = self.convr1(x)
        x = self._crop_concat(x, down_tensor)
        x = self.convr2(x)
        '''

        x = self.upSample(x)
        x = self._crop_concat(x, down_tensor)
        x = self.convr1(x)
        if self.drop_out: x = self.dropout_layer(x)
        x = self.convr2(x)
        if self.drop_out: x = self.dropout_layer(x)
        return x


class UNet_BN(nn.Module):
    def __init__(self):
        super(UNet_BN, self).__init__()
        # channels, height, width = in_shape

        # self.dropout = dropout

        self.model_code = ""        # variable to save model and main function when loaded

        self.down1 = StackEncoder(1, 64)
        self.down2 = StackEncoder(64, 128)
        self.down3 = StackEncoder(128, 256)
        self.down4 = StackEncoder(256, 512)

        self.center = nn.Sequential(
            # nn.Dropout2d(p=0.2),
            ConvBnRelu(512, 1024, kernel_size=(3, 3), stride=1, padding=0),
            nn.Dropout2d(p=0.2),
            ConvBnRelu(1024, 1024, kernel_size=(3, 3), stride=1, padding=0),
            nn.Dropout2d(p=0.2)
        )

        self.up1 = StackDecoder(in_channels=1024, out_channels=512, upsample_size=(64,64), padding = 1)
        self.up2 = StackDecoder(in_channels=512, out_channels=256, upsample_size=(128,128), padding = 1)
        self.up3 = StackDecoder(in_channels=256, out_channels=128, upsample_size=(256,256), padding = 1)
        self.up4 = StackDecoder(in_channels=128, out_channels=64, upsample_size=(512,512), padding = 1)

        # 1x1 convolution at the last layer
        # Different from the paper is the output size here
        self.output_seg_map = nn.Conv2d(64, 1, kernel_size=(1, 1), padding=0, stride=1)

        self._initialize_weights()

    def forward(self, x):
        x, x_trace1 = self.down1(x)
        x, x_trace2 = self.down2(x)
        x, x_trace3 = self.down3(x)
        x, x_trace4 = self.down4(x)
        x = self.center(x)
        x = self.up1(x, x_trace4)
        x = self.up2(x, x_trace3)
        x = self.up3(x, x_trace2)
        x = self.up4(x, x_trace1)

        x = self.output_seg_map(x)
        # x = F.sigmoid(x)
        # out = torch.squeeze(out, dim=1)
        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

# x = Variable(torch.randn(1,1,636,636))       # simulate input image
# net = UNet_BN()
# out = net(x)
# print("input: ", x)
# print("output: ", out)