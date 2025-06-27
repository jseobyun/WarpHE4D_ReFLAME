import os.path

import torch
import torch.nn as nn
import torchvision.models as tvm
from torch import Tensor
from typing import Dict
from warphead.utils import get_autocast_params

class VGG19(nn.Module):
    def __init__(self, pretrained=False, amp = False, amp_dtype = torch.float32) -> None:
        super().__init__()
        self.layers = nn.ModuleList(tvm.vgg19_bn(pretrained=pretrained).features[:40])
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x: Tensor):
        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(x.device, self.amp, self.amp_dtype)
        with torch.autocast(device_type=autocast_device, enabled=autocast_enabled, dtype = autocast_dtype):
            feats: Dict[int, Tensor] = {}
            scale = 1
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    feats[scale] = x
                    scale = scale*2
                x = layer(x)
            return feats

class Helper(nn.Module):
    def __init__(self, helper_features=None):
        super().__init__()
        self.helper_weights = None
        if helper_features is not None and os.path.exists(helper_features):
            self.helper_weights = torch.load(helper_features)

    def forward(self):
        return self.helper_weights

class Encoder(nn.Module):
    def __init__(self, cnn_kwargs = None, amp = False, dinov2_weights = None, amp_dtype = torch.float32):
        super().__init__()
        if dinov2_weights is None:
            dinov2_weights = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth", map_location="cpu")
        from .transformer import vit_large
        vit_kwargs = dict(img_size= 518,
            patch_size= 14,
            init_values = 1.0,
            ffn_layer = "mlp",
            block_chunks = 0,
        )

        dinov2_vitl14 = vit_large(**vit_kwargs).eval()
        dinov2_vitl14.load_state_dict(dinov2_weights)
        cnn_kwargs = cnn_kwargs if cnn_kwargs is not None else {}

        self.cnn = VGG19(**cnn_kwargs)
        self.amp = amp
        self.amp_dtype = amp_dtype
        if self.amp:
            dinov2_vitl14 = dinov2_vitl14.to(self.amp_dtype)
        self.dinov2_vitl14 = [dinov2_vitl14] # ugly hack to not show parameters to DDP
    
    def train(self, mode: bool = True):
        return self.cnn.train(mode)
    
    def forward(self, x):
        B,C,H,W = x.shape
        feature_pyramid = self.cnn(x)

        with torch.no_grad():
            if self.dinov2_vitl14[0].device != x.device:
                self.dinov2_vitl14[0] = self.dinov2_vitl14[0].to(x.device).to(self.amp_dtype)
            dinov2_features_16 = self.dinov2_vitl14[0].forward_features(x.to(self.amp_dtype))
            features_16 = dinov2_features_16['x_norm_patchtokens'].permute(0,2,1).reshape(B,1024,H//14, W//14)
            del dinov2_features_16
            feature_pyramid[16] = features_16

        return feature_pyramid