import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import clip
import open_clip
from .srm_filter_kernel import all_normalized_hpf_list
import numpy as np

import pandas as pd
import os
from pathlib import Path

class HPF(nn.Module):
  def __init__(self):
    super(HPF, self).__init__()

    #Load 30 SRM Filters
    all_hpf_list_5x5 = []

    for hpf_item in all_normalized_hpf_list:
      if hpf_item.shape[0] == 3:
        hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode='constant')

      all_hpf_list_5x5.append(hpf_item)

    hpf_weight = torch.Tensor(all_hpf_list_5x5).view(30, 1, 5, 5).contiguous()
    hpf_weight = torch.nn.Parameter(hpf_weight.repeat(1, 3, 1, 1), requires_grad=False)
   

    self.hpf = nn.Conv2d(3, 30, kernel_size=5, padding=2, bias=False)
    self.hpf.weight = hpf_weight


  def forward(self, input):

    output = self.hpf(input)

    return output



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=True):
        super(ResNet, self).__init__()

        self.inplanes = 64
        self.conv1 = nn.Conv2d(30, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)


        return x

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class AIDE_Model(nn.Module):

    def __init__(self, resnet_path, convnext_path, cache_csv_path=None):
        super(AIDE_Model, self).__init__()
        self.hpf = HPF()
        self.model_min = ResNet(Bottleneck, [3, 4, 6, 3])
        self.model_max = ResNet(Bottleneck, [3, 4, 6, 3])

        # CSV cache configuration
        self.cache_csv_path = cache_csv_path
        self.feature_cache = {}  # In-memory cache for faster lookup
        
        # Load existing cache if available
        if self.cache_csv_path and os.path.exists(self.cache_csv_path):
            self._load_cache()
       
        if resnet_path is not None:
            pretrained_dict = torch.load(resnet_path, map_location='cpu')
        
            model_min_dict = self.model_min.state_dict()
            model_max_dict = self.model_max.state_dict()
    
            for k in pretrained_dict.keys():
                if k in model_min_dict and pretrained_dict[k].size() == model_min_dict[k].size():
                    model_min_dict[k] = pretrained_dict[k]
                    model_max_dict[k] = pretrained_dict[k]
                else:
                    print(f"Skipping layer {k} because of size mismatch")
        
        self.fc = Mlp(2048 + 256 , 1024, 2)

        print("build model with convnext_xxl")
        self.openclip_convnext_xxl, _, _ = open_clip.create_model_and_transforms(
            "convnext_xxlarge", pretrained=convnext_path
        )

        self.openclip_convnext_xxl = self.openclip_convnext_xxl.visual.trunk
        self.openclip_convnext_xxl.head.global_pool = nn.Identity()
        self.openclip_convnext_xxl.head.flatten = nn.Identity()

        self.openclip_convnext_xxl.eval()
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.convnext_proj = nn.Sequential(
            nn.Linear(3072, 256),

        )
        for param in self.openclip_convnext_xxl.parameters():
            param.requires_grad = False

    def _load_cache(self):
        """Load cached features from CSV file"""
        try:
            df = pd.read_csv(self.cache_csv_path)
            for _, row in df.iterrows():
                image_path = row['image_path']
                # Convert string representation back to tensor
                feature_str = row['x_1_features']
                feature_list = [float(x) for x in feature_str.strip('[]').split(',')]
                feature_tensor = torch.tensor(feature_list).unsqueeze(0)  # Add batch dimension
                self.feature_cache[image_path] = feature_tensor
            print(f"Loaded {len(self.feature_cache)} cached features from {self.cache_csv_path}")
        except Exception as e:
            print(f"Error loading cache: {e}")
            self.feature_cache = {}

    def _get_cached_features(self, image_paths):
        """Get cached features for given image paths"""
        cached_features = []
        cache_mask = []  # Track which images have cached features
        
        for img_path in image_paths:
            if img_path and img_path in self.feature_cache:
                cached_features.append(self.feature_cache[img_path])
                cache_mask.append(True)
            else:
                cached_features.append(None)
                cache_mask.append(False)
                
        return cached_features, cache_mask

    

    def forward(self, x, image_paths=None, use_cache=False):

        b, t, c, h, w = x.shape

        x_minmin = x[:, 0] #[b, c, h, w]
        x_maxmax = x[:, 1]
        x_minmin1 = x[:, 2]
        x_maxmax1 = x[:, 3]
        x_minmin16 = x[:, 4]
        x_maxmax16 = x[:, 5]
        x_minmin1_16 = x[:, 6]
        x_maxmax1_16 = x[:, 7]
        # x_minmin64 = x[:, 8]
        # x_maxmax64 = x[:, 9]
        # x_minmin1_64 = x[:, 10]
        # x_maxmax1_64 = x[:, 11]
        tokens = x[:, 8]

        # Check cache first if image_paths provided and caching is enabled
        x_1_cached = None
        cache_mask = None
        
        if use_cache and image_paths is not None and self.cache_csv_path:
            cached_features, cache_mask = self._get_cached_features(image_paths)
            
            # If all features are cached, use them directly
            if all(cache_mask):
                x_1_cached = torch.cat(cached_features, dim=0).to(x.device)
                print(f"Using cached features for {len(image_paths)} images")

        x_minmin = self.hpf(x_minmin)
        x_maxmax = self.hpf(x_maxmax)
        x_minmin1 = self.hpf(x_minmin1)
        x_maxmax1 = self.hpf(x_maxmax1)

        x_minmin16 = self.hpf(x_minmin16)
        x_maxmax16 = self.hpf(x_maxmax16)
        x_minmin1_16 = self.hpf(x_minmin1_16)
        x_maxmax1_16 = self.hpf(x_maxmax1_16)

        # x_minmin64 = self.hpf(x_minmin64)
        # x_maxmax64 = self.hpf(x_maxmax64)
        # x_minmin1_64 = self.hpf(x_minmin1_64)
        # x_maxmax1_64 = self.hpf(x_maxmax1_64)

        with torch.no_grad():
            
            clip_mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073])
            clip_mean = clip_mean.to(tokens, non_blocking=True).view(3, 1, 1)
            clip_std = torch.Tensor([0.26862954, 0.26130258, 0.27577711])
            clip_std = clip_std.to(tokens, non_blocking=True).view(3, 1, 1)
            dinov2_mean = torch.Tensor([0.485, 0.456, 0.406]).to(tokens, non_blocking=True).view(3, 1, 1)
            dinov2_std = torch.Tensor([0.229, 0.224, 0.225]).to(tokens, non_blocking=True).view(3, 1, 1)

            local_convnext_image_feats = self.openclip_convnext_xxl(
                tokens * (dinov2_std / clip_std) + (dinov2_mean - clip_mean) / clip_std
            ) #[b, 3072, 8, 8]
            assert local_convnext_image_feats.size()[1:] == (3072, 8, 8)
            local_convnext_image_feats = self.avgpool(local_convnext_image_feats).view(tokens.size(0), -1)
            x_0 = self.convnext_proj(local_convnext_image_feats)

        x_min = self.model_min(x_minmin)
        x_max = self.model_max(x_maxmax)
        x_min1 = self.model_min(x_minmin1)
        x_max1 = self.model_max(x_maxmax1)

        x_min16 = self.model_min(x_minmin16)
        x_max16 = self.model_max(x_maxmax16)
        x_min1_16 = self.model_min(x_minmin1_16)
        x_max1_16 = self.model_max(x_maxmax1_16)

        # x_min64 = self.model_min(x_minmin64)
        # x_max64 = self.model_max(x_maxmax64)
        # x_min1_64 = self.model_min(x_minmin1_64)
        # x_max1_64 = self.model_max(x_maxmax1_64)

        # total = (x_min + x_max + x_min1 + x_max1) + (x_min16 + x_max16 + x_min1_16 + x_max1_16) + (x_min64 + x_max64 + x_min1_64 + x_max1_64) 

        total = (x_min + x_max + x_min1 + x_max1) + (x_min16 + x_max16 + x_min1_16 + x_max1_16)

        if use_cache:
            total = total + x_1_cached
            x_1 = total / 12

        else:
            x_1 = total / 8

        x = torch.cat([x_0, x_1], dim=1)

        x = self.fc(x)

        return x

def AIDE(resnet_path, convnext_path, cache_csv_path=None):
    model = AIDE_Model(resnet_path, convnext_path, cache_csv_path)
    return model

