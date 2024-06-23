import torch
import torch.nn as nn

class Residual_block2D(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])

        self.lrelu = nn.LeakyReLU(negative_slope=0.3)

        self.conv1 = nn.Conv2d(
            in_channels=nb_filts[0],
            out_channels=nb_filts[1],
            kernel_size=3,
            padding=1,
            stride=1,
        )

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(
            in_channels=nb_filts[1],
            out_channels=nb_filts[1],
            padding=1,
            kernel_size=3,
            stride=1,
        )

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(
                in_channels=nb_filts[0],
                out_channels=nb_filts[1],
                padding=0,
                kernel_size=1,
                stride=1,
            )
        else:
            self.downsample = False

        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.lrelu(out)
        else:
            out = x

        out = self.conv1(x)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        out = self.mp(out)
        return out

class SpecRNet(nn.Module):
    def __init__(self, d_args, **kwargs):
        super().__init__()

        self.device = kwargs.get("device", "cuda")

        self.first_bn = nn.BatchNorm2d(num_features=1)  # 입력 채널 수를 1로 설정
        self.selu = nn.SELU(inplace=True)
        self.block0 = nn.Sequential(
            Residual_block2D(nb_filts=[1, 16], first=True)  # 입력 채널 수를 1로 설정
        )
        self.block2 = nn.Sequential(Residual_block2D(nb_filts=[16, 32]))
        self.block4 = nn.Sequential(Residual_block2D(nb_filts=[32, 64]))
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc_attention0 = self._make_attention_fc(
            in_features=16, l_out_features=16  # 여기서 in_features를 16으로 맞춤
        )
        self.fc_attention2 = self._make_attention_fc(
            in_features=32, l_out_features=32  # 여기서 in_features를 32로 맞춤
        )
        self.fc_attention4 = self._make_attention_fc(
            in_features=64, l_out_features=64  # 여기서 in_features를 64로 맞춤
        )

        self.bn_before_gru = nn.BatchNorm2d(num_features=64)
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=d_args["gru_node"],
            num_layers=d_args["nb_gru_layer"],
            batch_first=True,
            bidirectional=True,
        )

        self.fc1_gru = nn.Linear(
            in_features=d_args["gru_node"] * 2, out_features=d_args["nb_fc_node"] * 2
        )

        self.fc2_gru = nn.Linear(
            in_features=d_args["nb_fc_node"] * 2,
            out_features=d_args["nb_classes"],
            bias=True,
        )

        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.first_bn(x)
        x = self.selu(x)

        x0 = self.block0(x)
        print("Shape after block0:", x0.shape)
        y0 = self.avgpool(x0).view(x0.size(0), -1)
        print("Shape after avgpool and view:", y0.shape)
        y0 = self.fc_attention0(y0)
        y0 = self.sig(y0).view(y0.size(0), y0.size(1), 1, 1)
        x = x0 * y0 + y0

        x = nn.MaxPool2d((2, 2))(x)  # 변경된 부분

        x2 = self.block2(x)
        print("Shape after block2:", x2.shape)
        y2 = self.avgpool(x2).view(x2.size(0), -1)
        y2 = self.fc_attention2(y2)
        y2 = self.sig(y2).view(y2.size(0), y2.size(1), 1, 1)
        x = x2 * y2 + y2

        x = nn.MaxPool2d((2, 2))(x)  # 변경된 부분

        x4 = self.block4(x)
        print("Shape after block4:", x4.shape)
        y4 = self.avgpool(x4).view(x4.size(0), -1)
        y4 = self.fc_attention4(y4)
        y4 = self.sig(y4).view(y4.size(0), y4.size(1), 1, 1)
        x = x4 * y4 + y4

        x = nn.MaxPool2d((2, 2))(x)  # 변경된 부분

        x = self.bn_before_gru(x)
        x = self.selu(x)
        x = x.squeeze(-2)
        x = x.permute(0, 2, 1)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc1_gru(x)
        x = self.fc2_gru(x)

        return x

    def _make_attention_fc(self, in_features, l_out_features):
        l_fc = []
        l_fc.append(nn.Linear(in_features=in_features, out_features=l_out_features))
        return nn.Sequential(*l_fc)
