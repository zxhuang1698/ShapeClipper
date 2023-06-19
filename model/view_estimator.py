import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as torch_F

class Bottleneck_Linear(nn.Module):

    def __init__(self, n_channels, zero_init=True):
        super().__init__()
        self.linear1 = nn.Conv2d(n_channels, n_channels, kernel_size=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.linear2 = nn.Conv2d(n_channels, n_channels, kernel_size=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(n_channels)
        self.relu = nn.ReLU(inplace=True)
        
        if zero_init:
            nn.init.constant_(self.bn2.weight, 0)

    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1)
        residual = x

        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        out = out.squeeze(-1).squeeze(-1)

        return out

class Estimator(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.dataset = opt.data.dataset
        self.feature_extractor = torchvision.models.resnet18(pretrained=True)
        n_features = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Identity()
        self.extr_head = nn.Sequential(
            Bottleneck_Linear(n_features)
        )
        self.size_head = nn.Sequential(
            Bottleneck_Linear(n_features)
        )
        self.perspect_head = nn.Sequential(
            Bottleneck_Linear(n_features)
        )
        
        self.extr_fc = nn.Linear(n_features, 6)
        self.size_fc = nn.Linear(n_features, 1)
        self.perspect_fc = nn.Linear(n_features, 1)
        
        # initialize extrinsic so elev and theta are zero
        # cos(0) = 1, sin(0) = 0, therefore the output after normalization should be [1, 0]
        torch.nn.init.constant_(self.extr_fc.weight[2:, :], 0.0)
        torch.nn.init.constant_(self.extr_fc.bias[2], 1.0)
        torch.nn.init.constant_(self.extr_fc.bias[3], 0.0)
        torch.nn.init.constant_(self.extr_fc.bias[4], 1.0)
        torch.nn.init.constant_(self.extr_fc.bias[5], 0.0)
        
        # initialize intrinsic so the scales are one
        torch.nn.init.constant_(self.size_fc.weight, 0.0)
        torch.nn.init.constant_(self.size_fc.bias, 0.0)
        torch.nn.init.constant_(self.perspect_fc.weight, 0.0)
        torch.nn.init.constant_(self.perspect_fc.bias, 0.0)

    def reset_scales(self):
        # initialize intrinsic so the scales are one
        torch.nn.init.constant_(self.size_fc.weight, 0.0)
        torch.nn.init.constant_(self.size_fc.bias, 0.0)
        torch.nn.init.constant_(self.perspect_fc.weight, 0.0)
        torch.nn.init.constant_(self.perspect_fc.bias, 0.0)

    def forward(self, inputs):
        feat = self.feature_extractor(inputs)
        # extrinsics
        feat_extr = self.extr_head(feat)
        trig_extr = self.extr_fc(feat_extr)
        output_a = trig_extr[:, :2]
        output_e = trig_extr[:, 2:4]
        output_t = trig_extr[:, 4:6]
        output_a = torch_F.normalize(output_a, dim=1, p=2)
        output_e = torch_F.normalize(output_e, dim=1, p=2)
        output_t = torch_F.normalize(output_t, dim=1, p=2)
        # scale_size is shrinking the size of the shape
        feat_size = self.size_head(feat)
        scale_size_raw = torch.tanh(self.size_fc(feat_size)).squeeze(-1)
        scale_size = 1 + scale_size_raw * self.opt.camera.size_range
        
        # scale_perspect control the perspective effect
        feat_perspect = self.perspect_head(feat)
        scale_perspect_raw = torch.tanh(self.perspect_fc(feat_perspect)).squeeze(-1)
        scale_perspect = 1 + scale_perspect_raw * self.opt.camera.perspect_range
        
        # convert to the scaling factor for focal and camera distance
        scale_focal = scale_perspect
        scale_dist = scale_size * scale_perspect
        
        return output_a, output_e, output_t, scale_focal, scale_dist