"""
Loss functions for IR-drop prediction and multi-task learning.

This file implements all loss modules used in the project, including:
  • Standard L1 and MSE losses with CircuitNet-style masking/reduction.
  • A combined L1 + Laplacian smoothness loss.
  • A multitask regression + classification loss for IR-drop hotspot prediction.

Contents
--------
1. Infrastructure
   • build_loss(...) :
        Dynamically constructs a loss class from config (opt dict),
        resolving constructor arguments safely.
   • masked_loss decorator :
        Wraps base loss functions so they support CircuitNet's mask +
        reduction semantics.
   • reduce_loss / mask_reduce_loss :
        Core utilities implementing reduction='mean'/'sum' and
        sample-wise normalization consistent with CircuitNet.

2. Base Losses
   • l1_loss / mse_loss :
        Elementwise losses (no reduction), automatically masked/reduced
        by the wrapper.
   • L1Loss :
        nn.Module wrapper around masked L1.
   • MSELoss :
        nn.Module wrapper around masked MSE.

3. Laplacian Regularization
   • _laplacian(x) :
        3x3 discrete Laplacian via depthwise conv, per-channel.
   • L1LapLoss :
        Combined loss = L1(pred,target) + lambda * L1(Lap(pred), Lap(target)).
        Supports:
            lap_mode = 'match'  → match Lap(pred) to Lap(target)
            lap_mode = 'pred'   → regularize Lap(pred) toward zero

4. Multi-task Learning
   • MultiTaskIRDropLoss :
        Joint loss for:
            - Regression head (masked L1 on IR-drop map)
            - Classification head (masked BCE on hotspot mask)
        Supports same reduction and sample_wise options as the base L1Loss.

All loss classes follow CircuitNet conventions:
  - Default loss_weight=100 to keep consistency with original MAVI/MAVIREC.
  - weight masks must match prediction shape.
  - No in-place reductions; masking logic is shared across all losses.
"""


import functools

import torch.nn as nn
import torch.nn.functional as F

import importlib
try:
    import utils.losses as losses
except Exception:
    # When running tests/imports outside of package context, fall back
    # to this module object so build_loss can still resolve classes.
    losses = importlib.import_module(__name__)


def build_loss(opt):
    # Resolve loss class by name and pass relevant constructor args (if provided)
    loss_type = opt.pop('loss_type')
    loss_cls = losses.__dict__[loss_type]
    # Only pass constructor args that our loss classes expect to avoid TypeError
    kwargs = {}
    for k in ['loss_weight', 'lap_weight', 'reduction', 'sample_wise', 'lap_mode']:
        if k in opt:
            kwargs[k] = opt.pop(k)
    return loss_cls(**kwargs)

__all__ = ['L1Loss', 'MSELoss', 'L1LapLoss', 'MultiTaskIRDropLoss']


def reduce_loss(loss, reduction):
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    if reduction_enum == 1:
        return loss.mean()

    return loss.sum()


def mask_reduce_loss(loss, weight=None, reduction='mean', sample_wise=False):
    if weight is not None:
        assert weight.dim() == loss.dim()
        assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    if weight is None or reduction == 'sum':
        loss = reduce_loss(loss, reduction)
    elif reduction == 'mean':
        if weight.size(1) == 1:
            weight = weight.expand_as(loss)
        eps = 1e-12

        if sample_wise:
            weight = weight.sum(dim=[1, 2, 3], keepdim=True)
            loss = (loss / (weight + eps)).sum() / weight.size(0)
        else:
            loss = loss.sum() / (weight.sum() + eps)

    return loss

def masked_loss(loss_func):
    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                sample_wise=False,
                **kwargs):
        loss = loss_func(pred, target, **kwargs)
        loss = mask_reduce_loss(loss, weight, reduction, sample_wise)
        return loss

    return wrapper

@masked_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@masked_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')

class L1Loss(nn.Module):
    def __init__(self, loss_weight=100.0, reduction='mean', sample_wise=False):
        super().__init__()

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * l1_loss(
            pred,
            target,
            weight,
            reduction=self.reduction,
            sample_wise=self.sample_wise)



class MSELoss(nn.Module):
    def __init__(self, loss_weight=100.0, reduction='mean', sample_wise=False):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * mse_loss(
            pred,
            target,
            weight,
            reduction=self.reduction,
            sample_wise=self.sample_wise)


# --- Laplacian-smoothness combined loss (L1 + Laplacian L1) ---
def _laplacian(x):
    import torch
    import torch.nn.functional as F

    # Accept [N, H, W] or [N, C, H, W]
    if x.dim() == 3:
        x = x.unsqueeze(1)  # [N,1,H,W]
    # 3x3 Laplacian kernel
    kernel = torch.tensor([[0., 1., 0.],
                           [1., -4., 1.],
                           [0., 1., 0.]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    # repeat per-channel for depthwise conv
    kernel = kernel.repeat(x.size(1), 1, 1, 1)  # [C,1,3,3]
    # depthwise conv to compute laplacian per channel
    lap = F.conv2d(x, kernel, padding=1, groups=x.size(1))
    return lap


class L1LapLoss(nn.Module):
    """Combined L1 loss + Laplacian L1 term.

    Args:
        loss_weight: global multiplier (keeps parity with existing losses).
        lap_weight: multiplier for the laplacian term (relative to base L1).
        reduction/sample_wise: passed to masked loss behaviour.
    """
    def __init__(self, loss_weight=100.0, lap_weight=1.0, reduction='mean', sample_wise=False, lap_mode='match'):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise
        self.lap_weight = lap_weight
        self.lap_mode = lap_mode

    def forward(self, pred, target, weight=None, **kwargs):
        # base L1 term (uses masked l1 implementation)
        base = l1_loss(pred, target, weight, reduction=self.reduction, sample_wise=self.sample_wise)

        # laplacian term behavior depends on lap_mode
        pred_lap = _laplacian(pred)
        target_lap = _laplacian(target) if target is not None else None
        # squeeze channel dim if present (we expect single-channel final maps)
        if pred_lap.dim() == 4 and pred_lap.size(1) == 1:
            pred_lap_s = pred_lap.squeeze(1)
            target_lap_s = target_lap.squeeze(1)
        else:
            # average across channels if multi-channel
            pred_lap_s = pred_lap.mean(dim=1)
            target_lap_s = target_lap.mean(dim=1)

        if self.lap_mode == 'match':
            lap_loss = l1_loss(pred_lap_s, target_lap_s, weight, reduction=self.reduction, sample_wise=self.sample_wise)
        else:
            # 'pred' mode: regularize prediction Laplacian towards zero
            zero = 0 * pred_lap_s
            lap_loss = l1_loss(pred_lap_s, zero, weight, reduction=self.reduction, sample_wise=self.sample_wise)

        return self.loss_weight * (base + self.lap_weight * lap_loss)

class MultiTaskIRDropLoss(nn.Module):
    """
    Multi-task IR-drop loss:
      - Regression: L1
      - Classification: BCE
    Both terms follow the same masking, reduction, and sample_wise semantics
    as the existing L1Loss implementation in CircuitNet.
    """

    def __init__(self, loss_weight=100.0, cls_weight=0.4,
                 reduction='mean', sample_wise=False):
        super().__init__()
        self.loss_weight = loss_weight
        self.cls_weight = cls_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, preds, targets, weight=None, **kwargs):
        """
        preds:   (reg_pred, cls_pred)
        targets: (reg_target, cls_target)

        reg_pred: [B,H,W]
        cls_pred: [B,H,W]   (sigmoid probability)
        """
        reg_pred, cls_pred = preds
        reg_target, cls_target = targets

        # -------------------------------------------------------
        # Regression: masked L1 (uses CircuitNet's l1_loss)
        # -------------------------------------------------------
        l1 = l1_loss(reg_pred, reg_target,
                     weight,
                     reduction=self.reduction,
                     sample_wise=self.sample_wise)

        # -------------------------------------------------------
        # Classification: masked BCE
        # -------------------------------------------------------
        # raw BCE without reduction
        bce_raw = F.binary_cross_entropy(cls_pred, cls_target,
                                         reduction='none')

        # mask + reduce identically to regression
        bce = mask_reduce_loss(bce_raw, weight,
                               reduction=self.reduction,
                               sample_wise=self.sample_wise)

        # -------------------------------------------------------
        # Final combined loss
        # -------------------------------------------------------
        total = l1 + self.cls_weight * bce

        return self.loss_weight * total


