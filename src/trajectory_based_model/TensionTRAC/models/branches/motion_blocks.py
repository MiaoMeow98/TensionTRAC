"""2D trajectory motion blocks used by the pretrained Pointformer checkpoint."""

from collections import OrderedDict

import torch
import torch.nn as nn
from einops import rearrange


class LayerNorm(nn.LayerNorm):
    """LayerNorm that preserves the input dtype."""

    def forward(self, input: torch.Tensor):
        orig_type = input.dtype
        ret = super().forward(input.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


def cross_patch_motion_v1_allneighbors(point_trajs_gt_coord, point_trajs_visibility_mask):
    batch_size, num_points, temporal_len, point_dim = point_trajs_gt_coord.shape
    assert point_trajs_visibility_mask.shape == (batch_size, num_points, temporal_len)

    nan_indices = torch.isnan(point_trajs_gt_coord)
    point_trajs_gt_coord = torch.nan_to_num(point_trajs_gt_coord, nan=0.0)
    point_trajs_visibility_mask[nan_indices[..., 0]] = 0

    tensor_centers = point_trajs_gt_coord.unsqueeze(3)
    tensor_neighbors = point_trajs_gt_coord.permute(0, 2, 1, 3).unsqueeze(1)
    distances_relative = tensor_centers - tensor_neighbors

    vis_mask_centers = point_trajs_visibility_mask.unsqueeze(3)
    vis_mask_neighbors = point_trajs_visibility_mask.permute(0, 2, 1).unsqueeze(1)
    vis_mask_pair = vis_mask_centers * vis_mask_neighbors
    distances_relative = distances_relative * vis_mask_pair.unsqueeze(-1)

    feature_dim = num_points * point_dim
    return distances_relative.reshape(batch_size, num_points, temporal_len, feature_dim)


class CrossMotionModule(nn.Module):
    """Legacy cross-trajectory motion encoder."""

    def __init__(self, out_feature_dim=768, num_patches=256, in_fea_dim_crossmotion=512):
        super().__init__()
        self.num_patches = num_patches
        self.ln2 = LayerNorm(out_feature_dim)
        self.fc2 = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(in_fea_dim_crossmotion, 4 * out_feature_dim)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(4 * out_feature_dim, out_feature_dim)),
                ]
            )
        )

    def forward(self, point_trajs_gt_coord, point_trajs_visibility_mask):
        point_trajs_gt_coord = rearrange(point_trajs_gt_coord, "b t m d -> b m t d")
        batch_size, num_points, temporal_len, _ = point_trajs_gt_coord.shape
        point_trajs_visibility_mask = rearrange(point_trajs_visibility_mask, "b t m -> b m t")
        assert point_trajs_visibility_mask.shape == (batch_size, num_points, temporal_len)
        assert point_trajs_visibility_mask.max() <= 1.0
        assert point_trajs_visibility_mask.min() >= 0.0

        cross_motion = cross_patch_motion_v1_allneighbors(
            point_trajs_gt_coord,
            point_trajs_visibility_mask,
        )
        cross_motion = self.ln2(self.fc2(cross_motion))
        return rearrange(cross_motion, "b m t d -> b t m d")


class HODMotionModule(nn.Module):
    """Histogram-of-displacements intra-trajectory motion encoder."""

    def __init__(self, out_feature_dim=768, num_patches=256, in_feature_dim=32):
        super().__init__()
        self.num_patches = num_patches
        self.ln2 = LayerNorm(out_feature_dim)
        self.fc2 = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(in_feature_dim, 4 * out_feature_dim)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(4 * out_feature_dim, out_feature_dim)),
                ]
            )
        )

    def forward(self, hod_feat):
        hod_motion = self.ln2(self.fc2(hod_feat))
        return hod_motion.permute(0, 2, 1, 3)
