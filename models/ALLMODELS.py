"""
This file defines all model architectures used in the project, including the
baseline MAVI network, several enhanced variants, and a multitask extension.

Contents
--------
1. Utility
   - load_state_dict : Lightweight loader compatible with CircuitNet checkpoints.

2. Basic Blocks
   - DoubleConv3d / DoubleConv2d : 3D/2D convolutional blocks with BN + ReLU.
   - Down : 3D max-pool + DoubleConv3d.
   - Up   : 2D upsampling + skip-connection fusion + DoubleConv2d.
   - OutConv : Final 1×1 conv used for regression logits.

3. Baseline Models
   - MAVI : Original MAVIREC-style architecture (3D encoder, 2D decoder),
            with static-feature modulation and channel-summation head.
   - MAVI_with_SE : MAVI augmented with Squeeze-and-Excitation blocks.

4. Temporal Attention
   - TemporalAttention : Learnable aggregation over the T dimension.
   - MAVI_TemporalAttention : MAVI using temporal attention instead of mean-over-time.

5. Static-Feature FiLM Variants
   - MAVIStaticFiLM : Uses static features as conditioning via FiLM (gamma,beta).

6. Temporal-Only + FiLM + TA Variant
   - MAVI_TA_StaticFiLM_TemporalOnly :
        - Temporal-only encoder
        - Temporal attention at all scales
        - Static-feature FiLM modulation
        - MAVIREC-style regression head

7. Multitask Extension
   - MAVI_TA_StaticFiLM_TemporalOnly_MultiTask :
        - Same backbone as the above model
        - Regression head identical to MAVI/MAVIREC
        - Additional classification head operating on the FiLM-modulated features

All models follow CircuitNet-compatible initialization patterns (Kaiming, BN init),
and include init_weights() hooks for loading pretrained checkpoints.
"""

import torch
import torch.nn as nn

from mmcv.cnn import constant_init, kaiming_init
from mmcv.utils.parrots_wrapper import _BatchNorm

import torch
import torch.nn as nn
import torch.nn.functional as F


from collections import OrderedDict

def load_state_dict(module, state_dict, strict=False, logger=None):
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None

    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    if len(err_msg) > 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)
    return missing_keys


class DoubleConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)



class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
            DoubleConv3d(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv2d(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv2d(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class MAVI(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 out_channels=4,
                 bilinear=False,
                 **kwargs):
        super(MAVI, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv3d(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1

        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_channels)


    def forward(self, x):
        x_in = x[:, :, :self.out_channels, :, :]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4.mean(dim=2), x3.mean(dim=2))
        x = self.up2(x, x2.mean(dim=2))
        x = self.up3(x, x1.mean(dim=2))
        logits = self.outc(x)

        # logits = x_in.squeeze(1)*logits
        logits = x_in.squeeze(1)*logits
        return torch.sum(logits, dim=1)

    def init_weights(self, pretrained=None, strict=True, **kwargs):
        if isinstance(pretrained, str):
            new_dict = OrderedDict()
            weight = torch.load(pretrained, map_location='cpu')['state_dict']
            for k in weight.keys():
                new_dict[k] = weight[k]
            load_state_dict(self, new_dict, strict=strict, logger=None)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    constant_init(m.weight, 1)
                    constant_init(m.bias, 0)

                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
        else:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')


class SEBlock(nn.Module):
    """Channel Squeeze-and-Excitation block."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.size()
        z = x.mean(dim=(2, 3))          # Global average pool -> [B, C]
        w = self.fc(z).view(B, C, 1, 1) # Channel attention weights
        return x * w                    # Scale channels

class MAVI_with_SE(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 out_channels=4,
                 bilinear=False,
                 **kwargs):
        super(MAVI_with_SE, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        # Encoder (same as MAVI)
        self.inc = DoubleConv3d(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        factor = 2 if bilinear else 1

        # Decoder
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_channels)

        # ---- Add SE blocks in bottleneck + skip path ----
        self.se_bottleneck = SEBlock(512)
        self.se3 = SEBlock(256)
        self.se2 = SEBlock(128)
        self.se1 = SEBlock(64)

    def forward(self, x):
        # x: [B, 1, T, H, W]
        x_in = x[:, :, :self.out_channels, :, :]  # static feature selection

        # Encoder path
        x1 = self.inc(x)         # [B,64,T,H,W]
        x2 = self.down1(x1)      # [B,128,T,H/2,W/2]
        x3 = self.down2(x2)      # [B,256,T,H/4,W/4]
        x4 = self.down3(x3)      # [B,512,T,H/8,W/8]

        # Temporal collapse (mean over time)
        x4_2d = x4.mean(dim=2)   # [B,512,H/8,W/8]
        x3_2d = x3.mean(dim=2)
        x2_2d = x2.mean(dim=2)
        x1_2d = x1.mean(dim=2)

        # Apply SE blocks
        x4_2d = self.se_bottleneck(x4_2d)
        x3_2d = self.se3(x3_2d)
        x2_2d = self.se2(x2_2d)
        x1_2d = self.se1(x1_2d)

        # Decoder path
        x = self.up1(x4_2d, x3_2d)
        x = self.up2(x,      x2_2d)
        x = self.up3(x,      x1_2d)

        logits = self.outc(x)               # [B, out_channels, H, W]

        # Multiply by static feature (CircuitNet design)
        logits = x_in.squeeze(1) * logits   # elementwise modulator
        return torch.sum(logits, dim=1)     # final IR-drop map
    def init_weights(self, pretrained=None, strict=True, **kwargs):
        if isinstance(pretrained, str):
            new_dict = OrderedDict()
            weight = torch.load(pretrained, map_location='cpu')['state_dict']
            for k in weight.keys():
                new_dict[k] = weight[k]
            load_state_dict(self, new_dict, strict=strict, logger=None)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    constant_init(m.weight, 1)
                    constant_init(m.bias, 0)

                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
        else:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')


class TemporalAttention(nn.Module):
    """
    Simple temporal attention over the T dimension.

    Input:  x [B, C, T, H, W]
    Output: y [B, C, H, W]   (attended aggregation over T)

    Mechanism:
      - Global average pool over H,W -> [B, C, T]
      - Per-time-step MLP over channels -> score_t
      - Softmax over T to get alpha_t
      - Weighted sum over T of the original x
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape

        # [B, C, T, H, W] -> [B, C, T] via spatial GAP
        s = x.mean(dim=(3, 4))          # [B, C, T]
        s = s.permute(0, 2, 1)          # [B, T, C]

        # Per-time MLP -> scalar score for each t
        h = F.relu(self.fc1(s))         # [B, T, hidden]
        scores = self.fc2(h)            # [B, T, 1]
        alpha = torch.softmax(scores, dim=1)  # [B, T, 1]

        # Reshape alphas to broadcast over C,H,W and aggregate
        alpha = alpha.permute(0, 2, 1)              # [B, 1, T]
        alpha = alpha.view(B, 1, T, 1, 1)           # [B, 1, T, 1, 1]

        # Weighted sum over time
        out = (x * alpha).sum(dim=2)                # [B, C, H, W]
        return out


# -------------------------------------------------------------------
# MAVI with temporal attention
# -------------------------------------------------------------------

class MAVI_TemporalAttention(nn.Module):
    """
    MAVI variant that replaces mean-over-time with learnable temporal attention.

    Input:  x [B, 1, T, H, W]
    Output: IR-drop map [B, H, W]
    """
    def __init__(self,
                 in_channels=1,
                 out_channels=4,
                 bilinear=False,
                 **kwargs):
        super(MAVI_TemporalAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        # Encoder (same as original MAVI)
        self.inc   = DoubleConv3d(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        factor = 2 if bilinear else 1

        # Decoder (same as original MAVI)
        self.up1  = Up(512, 256 // factor, bilinear)
        self.up2  = Up(256, 128 // factor, bilinear)
        self.up3  = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_channels)

        # Temporal attention for each encoder scale
        self.ta1 = TemporalAttention(64)
        self.ta2 = TemporalAttention(128)
        self.ta3 = TemporalAttention(256)
        self.ta4 = TemporalAttention(512)

    def forward(self, x):
        # x: [B, 1, T, H, W]
        # static “mask” features (kept exactly as in original MAVI)
        x_in = x[:, :, :self.out_channels, :, :]  # [B,1,out_ch,H,W]

        # Encoder with 3D convs
        x1 = self.inc(x)        # [B, 64,  T,   H,    W]
        x2 = self.down1(x1)     # [B, 128, T,   H/2,  W/2]
        x3 = self.down2(x2)     # [B, 256, T,   H/4,  W/4]
        x4 = self.down3(x3)     # [B, 512, T,   H/8,  W/8]

        # --- Temporal attention instead of .mean(dim=2) ---
        x1_2d = self.ta1(x1)    # [B, 64,  H,   W]
        x2_2d = self.ta2(x2)    # [B, 128, H/2, W/2]
        x3_2d = self.ta3(x3)    # [B, 256, H/4, W/4]
        x4_2d = self.ta4(x4)    # [B, 512, H/8, W/8]

        # Decoder path (unchanged)
        x = self.up1(x4_2d, x3_2d)
        x = self.up2(x,      x2_2d)
        x = self.up3(x,      x1_2d)

        logits = self.outc(x)               # [B, out_channels, H, W]

        # Multiply by static feature mask, then sum channels (same as MAVI)
        logits = x_in.squeeze(1) * logits   # [B, out_channels, H, W]
        ir_map = torch.sum(logits, dim=1)   # [B, H, W]

        return ir_map

    def init_weights(self, pretrained=None, strict=True, **kwargs):
        """
        Same init API as original MAVI; relies on load_state_dict,
        constant_init, kaiming_init, _BatchNorm already defined in this file.
        """
        if isinstance(pretrained, str):
            new_dict = OrderedDict()
            weight = torch.load(pretrained, map_location='cpu')['state_dict']
            for k in weight.keys():
                new_dict[k] = weight[k]
            load_state_dict(self, new_dict, strict=strict, logger=None)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    constant_init(m.weight, 1)
                    constant_init(m.bias, 0)

                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
        else:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')

class MAVIStaticFiLM(nn.Module):
    """
    MAVI variant that encodes static features and uses them to FiLM-modulate
    the decoder features, while still using the MAVIREC-style regression head.

    Assumes:
      - Input x: [B, 1, T, H, W]
      - First `out_channels` slices along T are static maps.
    """
    def __init__(self,
                 in_channels=1,
                 out_channels=4,
                 bilinear=False,
                 **kwargs):
        super(MAVIStaticFiLM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        # === Original 3D encoder ===
        self.inc = DoubleConv3d(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1

        # === Original 2D decoder ===
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_channels)

        # === NEW: static-feature encoder → FiLM parameters (γ, β) ===
        # Treat static maps as 2D channels: [B, C_static, H, W] with C_static = out_channels.
        # static_encoder outputs 2*64 channels which we split into (γ, β), each 64.
        self.static_encoder = nn.Sequential(
            DoubleConv2d(out_channels, 64),
            nn.Conv2d(64, 2 * 64, kernel_size=1)   # -> [B, 128, H, W] = [γ||β]
        )

    def forward(self, x):
        """
        x: [B, 1, T, H, W]
        First `out_channels` slices along T are static features.
        Remaining slices are temporal / dynamic maps.
        """
        # --- Static branch ---
        # x_static: [B, 1, C_static, H, W]
        x_static = x[:, :, :self.out_channels, :, :]
        # as 2D channels: [B, C_static, H, W]
        static_2d = x_static.squeeze(1)

        # static_encoder → FiLM parameters
        film_params = self.static_encoder(static_2d)   # [B, 128, H, W]
        gamma, beta = torch.chunk(film_params, 2, dim=1)  # each [B, 64, H, W]

        # --- Original 3D encoder / 2D decoder on full sequence ---
        # You can keep feeding the full x (static + temporal) into the 3D encoder
        # as in the original MAVI.
        x1 = self.inc(x)           # [B, 64, T, H, W]
        x2 = self.down1(x1)        # [B, 128, T, H/2, W/2]
        x3 = self.down2(x2)        # [B, 256, T, H/4, W/4]
        x4 = self.down3(x3)        # [B, 512, T, H/8, W/8]

        # Temporal averaging (same as original MAVI)
        x = self.up1(x4.mean(dim=2), x3.mean(dim=2))   # [B, 256, H/4, W/4]
        x = self.up2(x,              x2.mean(dim=2))   # [B, 128, H/2, W/2]
        x = self.up3(x,              x1.mean(dim=2))   # [B, 64,  H,   W]

        # --- FiLM modulation by static features ---
        # x:    [B, 64, H, W]
        # γ, β: [B, 64, H, W]
        x = gamma * x + beta

        # --- Regression head as in MAVIREC/MAVI implementation ---
        logits = self.outc(x)                        # [B, C_static, H, W]
        # Use the raw static maps as F_k(x)
        logits = x_static.squeeze(1) * logits        # [B, C_static, H, W]
        return torch.sum(logits, dim=1)              # [B, H, W]

    def init_weights(self, pretrained=None, strict=True, **kwargs):
        if isinstance(pretrained, str):
            new_dict = OrderedDict()
            weight = torch.load(pretrained, map_location='cpu')['state_dict']
            for k in weight.keys():
                new_dict[k] = weight[k]
            load_state_dict(self, new_dict, strict=strict, logger=None)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    constant_init(m.weight, 1)
                    constant_init(m.bias, 0)

                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
        else:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')

class MAVI_TA_StaticFiLM_TemporalOnly(nn.Module):
    """
    MAVI with:
      * Temporal-only encoder
      * Temporal attention at each scale
      * Static-feature FiLM modulation
      * MAVIREC-style regression head
    """
    def __init__(self,
                 in_channels=1,
                 out_channels=4,
                 bilinear=False,
                 **kwargs):
        super().__init__()

        self.out_channels = out_channels
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        # -------------------------------------------------------------
        # Static-feature encoder → FiLM (gamma, beta)
        # -------------------------------------------------------------
        self.static_encoder = nn.Sequential(
            DoubleConv2d(out_channels, 64),
            nn.Conv2d(64, 128, kernel_size=1)   # outputs 128 = 64 γ + 64 β
        )

        # -------------------------------------------------------------
        # Temporal encoder (3D)
        # -------------------------------------------------------------
        self.inc   = DoubleConv3d(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        # -------------------------------------------------------------
        # Temporal attention modules
        # -------------------------------------------------------------
        self.ta1 = TemporalAttention(64)
        self.ta2 = TemporalAttention(128)
        self.ta3 = TemporalAttention(256)
        self.ta4 = TemporalAttention(512)

        # -------------------------------------------------------------
        # Decoder (2D)
        # -------------------------------------------------------------
        self.up1  = Up(512, 256 // factor, bilinear)
        self.up2  = Up(256, 128 // factor, bilinear)
        self.up3  = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_channels)


    # -------------------------------------------------------------------
    # Forward pass
    # -------------------------------------------------------------------
    def forward(self, x):
        """
        x: [B, 1, T, H, W]
        First out_channels slices are static maps.
        Remaining slices are the temporal sequence.
        """
        B, _, T, H, W = x.shape

        # Separate static vs temporal
        x_static = x[:, :, :self.out_channels, :, :]          # [B,1,Cs,H,W]
        x_dyn    = x[:, :, self.out_channels:, :, :]          # [B,1,T_dyn,H,W]
        static_2d = x_static.squeeze(1)                        # [B,Cs,H,W]

        # --- Static-feature encoder → FiLM ---
        film_params = self.static_encoder(static_2d)   # [B,128,H,W]
        gamma, beta = torch.chunk(film_params, 2, dim=1)

        # --- Temporal encoder ---
        x1 = self.inc(x_dyn)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # --- Temporal attention ---
        x1_2d = self.ta1(x1)
        x2_2d = self.ta2(x2)
        x3_2d = self.ta3(x3)
        x4_2d = self.ta4(x4)

        # --- Decoder ---
        x = self.up1(x4_2d, x3_2d)
        x = self.up2(x,      x2_2d)
        x = self.up3(x,      x1_2d)

        # --- FiLM modulation ---
        x = gamma * x + beta

        # --- Regression head ---
        logits = self.outc(x)             # [B,Cs,H,W]
        logits = static_2d * logits
        return logits.sum(dim=1)          # [B,H,W]


    # -------------------------------------------------------------------
    # init_weights (matching CircuitNet style)
    # -------------------------------------------------------------------
    def init_weights(self, pretrained=None, strict=False, **kwargs):
        """
        pretrained: path to a .pth file
        strict=False is recommended, because this model adds new layers
        that are not present in original MAVI checkpoints.
        """
        if isinstance(pretrained, str):
            # Load pretrained weights
            state = torch.load(pretrained, map_location="cpu")

            # Allow both raw dict and {'state_dict': {...}} formats
            if "state_dict" in state:
                state = state["state_dict"]

            new_state = {}
            for k, v in state.items():
                # Strip "module." if checkpoint was from DDP
                new_k = k.replace("module.", "")
                new_state[new_k] = v

            self.load_state_dict(new_state, strict=strict)
            print(f"Loaded pretrained weights ({'strict' if strict else 'non-strict'}) from {pretrained}")
            return

        # -------------------------------------------------------------
        # Fresh initialization
        # -------------------------------------------------------------
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        print("Initialized all weights with Kaiming initialization.")

class MAVI_TA_StaticFiLM_TemporalOnly_MultiTask(nn.Module):
    """
    Multitask extension of MAVI_TA_StaticFiLM_TemporalOnly:
      - Body (static FiLM, temporal encoder, temporal attention, decoder)
        is identical to MAVI_TA_StaticFiLM_TemporalOnly.
      - Regression head is EXACTLY the same MAVIREC-style head.
      - Added classification head on the same FiLM-modulated features.
    """

    def __init__(self,
                 in_channels=1,
                 out_channels=4,
                 bilinear=False,
                 **kwargs):
        super().__init__()

        self.out_channels = out_channels
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        # -------------------------------------------------------------
        # Static-feature encoder → FiLM (gamma, beta)
        # (identical to MAVI_TA_StaticFiLM_TemporalOnly)
        # -------------------------------------------------------------
        self.static_encoder = nn.Sequential(
            DoubleConv2d(out_channels, 64),
            nn.Conv2d(64, 2 * 64, kernel_size=1)   # -> [B, 128, H, W] = [γ||β]
        )

        # -------------------------------------------------------------
        # Temporal encoder (3D)  -- identical
        # -------------------------------------------------------------
        self.inc   = DoubleConv3d(in_channels, 64)
        self.down1 = Down(64,   128)
        self.down2 = Down(128,  256)
        self.down3 = Down(256,  512)

        # -------------------------------------------------------------
        # Temporal attention modules  -- identical
        # -------------------------------------------------------------
        self.ta1 = TemporalAttention(64)
        self.ta2 = TemporalAttention(128)
        self.ta3 = TemporalAttention(256)
        self.ta4 = TemporalAttention(512)

        # -------------------------------------------------------------
        # Decoder (2D)  -- identical
        # -------------------------------------------------------------
        self.up1  = Up(512, 256 // factor, bilinear)
        self.up2  = Up(256, 128 // factor, bilinear)
        self.up3  = Up(128, 64,           bilinear)
        self.outc = OutConv(64, out_channels)   # same regression head conv

        # -------------------------------------------------------------
        # NEW: classification head (extra 1×1 conv)
        # -------------------------------------------------------------
        # Operates on the same FiLM-modulated feature map [B,64,H,W].
        self.cls_head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        """
        x: [B, 1, T, H, W]
        First out_channels slices are static maps.
        Remaining slices are temporal sequence.
        Returns:
            reg_map : [B, H, W]  (same as single-task model)
            cls_map : [B, H, W]  (sigmoid probabilities)
        """
        B, _, T, H, W = x.shape

        # -----------------------------
        # Split static vs temporal
        # -----------------------------
        x_static = x[:, :, :self.out_channels, :, :]   # [B,1,Cs,H,W]
        x_dyn    = x[:, :, self.out_channels:, :, :]   # [B,1,T_dyn,H,W]
        static_2d = x_static.squeeze(1)                # [B,Cs,H,W]

        # -----------------------------
        # Static-feature encoder → FiLM
        # -----------------------------
        film_params = self.static_encoder(static_2d)   # [B,128,H,W]
        gamma, beta = torch.chunk(film_params, 2, dim=1)  # [B,64,H,W] each

        # -----------------------------
        # Temporal encoder (3D)
        # -----------------------------
        x1 = self.inc(x_dyn)        # [B,64,T,H,W]
        x2 = self.down1(x1)         # [B,128,T,H/2,W/2]
        x3 = self.down2(x2)         # [B,256,T,H/4,W/4]
        x4 = self.down3(x3)         # [B,512,T,H/8,W/8]

        # -----------------------------
        # Temporal attention (collapse T)
        # -----------------------------
        x1_2d = self.ta1(x1)        # [B,64,H,  W]
        x2_2d = self.ta2(x2)        # [B,128,H/2,W/2]
        x3_2d = self.ta3(x3)        # [B,256,H/4,W/4]
        x4_2d = self.ta4(x4)        # [B,512,H/8,W/8]

        # -----------------------------
        # Decoder
        # -----------------------------
        x_dec = self.up1(x4_2d, x3_2d)
        x_dec = self.up2(x_dec, x2_2d)
        x_dec = self.up3(x_dec, x1_2d)   # [B,64,H,W]

        # -----------------------------
        # FiLM modulation (identical)
        # -----------------------------
        x_film = gamma * x_dec + beta    # [B,64,H,W]

        # -----------------------------
        # Regression head (IDENTICAL)
        # -----------------------------
        reg_logits = self.outc(x_film)              # [B,Cs,H,W]
        reg_logits = static_2d * reg_logits         # [B,Cs,H,W]
        reg_map    = reg_logits.sum(dim=1)          # [B,H,W]

        # -----------------------------
        # Classification head (NEW)
        # -----------------------------
        cls_logits = self.cls_head(x_film).squeeze(1)  # [B,H,W]
        cls_map    = torch.sigmoid(cls_logits)         # probabilistic mask

        return reg_map, cls_map

    def init_weights(self, pretrained=None, strict=False, **kwargs):
        """
        Init compatible with loading single-task checkpoints:
        shared layers (up to outc) will load the same weights;
        cls_head will be randomly initialized.
        """
        if isinstance(pretrained, str):
            state = torch.load(pretrained, map_location="cpu")
            if "state_dict" in state:
                state = state["state_dict"]

            new_state = {}
            for k, v in state.items():
                new_k = k.replace("module.", "")
                new_state[new_k] = v

            self.load_state_dict(new_state, strict=strict)
            print(f"Loaded pretrained weights ({'strict' if strict else 'non-strict'}) from {pretrained}")
            return

        # Fresh init
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        print("Initialized all weights with Kaiming initialization.")
