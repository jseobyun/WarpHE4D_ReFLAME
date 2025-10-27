import os
import math
import torch
import pickle
import trimesh
import numpy as np
import open3d as o3d
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import minimize
from ext.dwpose import DwposeDetector

@torch.no_grad()
def cls_to_flow_refine(cls):
    B, C, H, W = cls.shape
    device = cls.device
    res = round(math.sqrt(C))
    G = torch.meshgrid(
        *[torch.linspace(-1 + 1 / res, 1 - 1 / res, steps=res, device=device) for _ in range(2)],
        indexing='ij'
    )
    G = torch.stack([G[1], G[0]], dim=-1).reshape(C, 2)
    # FIXME: below softmax line causes mps to bug, don't know why.
    if device.type == 'mps':
        cls = cls.log_softmax(dim=1).exp()
    else:
        cls = cls.softmax(dim=1)
    mode = cls.max(dim=1).indices

    index = torch.stack((mode - 1, mode, mode + 1, mode - res, mode + res), dim=1).clamp(0, C - 1).long()
    neighbours = torch.gather(cls, dim=1, index=index)[..., None]
    flow = neighbours[:, 0] * G[index[:, 0]] + neighbours[:, 1] * G[index[:, 1]] + neighbours[:, 2] * G[
        index[:, 2]] + neighbours[:, 3] * G[index[:, 3]] + neighbours[:, 4] * G[index[:, 4]]
    tot_prob = neighbours.sum(dim=1)
    flow = flow / tot_prob
    return flow


def local_correlation(
        feature0,
        feature1,
        local_radius,
        padding_mode="zeros",
        flow=None,
        sample_mode="bilinear",
):
    r = local_radius
    K = (2 * r + 1) ** 2
    B, c, h, w = feature0.size()
    corr = torch.empty((B, K, h, w), device=feature0.device, dtype=feature0.dtype)
    if flow is None:
        # If flow is None, assume feature0 and feature1 are aligned
        coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=feature0.device),
                torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=feature0.device),
            ),
            indexing='ij'
        )
        coords = torch.stack((coords[1], coords[0]), dim=-1)[
            None
        ].expand(B, h, w, 2)
    else:
        coords = flow.permute(0, 2, 3, 1)  # If using flow, sample around flow target.
    local_window = torch.meshgrid(
        (
            torch.linspace(-2 * local_radius / h, 2 * local_radius / h, 2 * r + 1, device=feature0.device),
            torch.linspace(-2 * local_radius / w, 2 * local_radius / w, 2 * r + 1, device=feature0.device),
        ),
        indexing='ij'
    )
    local_window = torch.stack((local_window[1], local_window[0]), dim=-1)[
        None
    ].expand(1, 2 * r + 1, 2 * r + 1, 2).reshape(1, (2 * r + 1) ** 2, 2)
    for _ in range(B):
        with torch.no_grad():
            local_window_coords = (coords[_, :, :, None] + local_window[:, None, None]).reshape(1, h,
                                                                                                w * (2 * r + 1) ** 2, 2)
            window_feature = F.grid_sample(
                feature1[_:_ + 1], local_window_coords, padding_mode=padding_mode, align_corners=False,
                mode=sample_mode,  #
            )
            window_feature = window_feature.reshape(c, h, w, (2 * r + 1) ** 2)
        corr[_] = (feature0[_, ..., None] / (c ** .5) * window_feature).sum(dim=0).permute(2, 0, 1)
    return corr


def get_grid(b, h, w, device):
    grid = torch.meshgrid(
        *[
            torch.linspace(-1 + 1 / n, 1 - 1 / n, n, device=device)
            for n in (b, h, w)
        ],
        indexing='ij'
    )
    grid = torch.stack((grid[2], grid[1]), dim=-1).reshape(b, h, w, 2)
    return grid


def get_autocast_params(device=None, enabled=False, dtype=None):
    if device is None:
        autocast_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        # strip :X from device
        autocast_device = str(device).split(":")[0]
    if 'cuda' in str(device):
        out_dtype = dtype
        enabled = True
    else:
        out_dtype = torch.bfloat16
        enabled = False
        # mps is not supported
        autocast_device = "cpu"
    return autocast_device, enabled, out_dtype


def check_rgb(im):
    if im.mode != "RGB":
        raise NotImplementedError("Can't handle non-RGB images")

