import torch
import torch.nn as nn
from einops import rearrange

import TensionTRAC.utils.logging as logging
from TensionTRAC.models.fusion.attention import (
    CrossAttentionEncoderLayer,
    CrossAttentionTransformerEncoder,
    SelfAttention,
)

logger = logging.get_logger(__name__)


class FusionModule(nn.Module):
    """Fuse DINO point tokens with intra- and cross-trajectory motion tokens."""

    def __init__(
        self,
        fusion_type,
        semantic_feat_dim,
        use_hod_motion_module=True,
        hod_motion_feat_dim=768,
        use_cross_motion_module=True,
        cross_motion_feat_dim=768,
        use_semantic_features=True,
        output_dim=None,
        self_attention_num_heads=4,
        self_attention_dropout_rate=0.0,
        self_attention_use_ffn=False,
        self_attention_use_pos_encoding=False,
        self_attention_pos_encoding_type="fixed",
        self_attention_across_dim="spatial",
        self_attention_pos_encoding_across_dim="same",
        cross_attention_num_heads=4,
        cross_attention_num_layers=2,
        cross_attention_use_ffn=False,
        cross_attention_along_dim="spatial",
        gated_fusion_hidden_dim=128,
        gated_fusion_output_dim=768,
        final_fusion_type="concat",
        self_cross_attention_use_ffn=False,
    ):
        super().__init__()
        del (
            self_attention_use_pos_encoding,
            self_attention_pos_encoding_type,
            self_attention_pos_encoding_across_dim,
            gated_fusion_hidden_dim,
            self_cross_attention_use_ffn,
        )
        self.visual_feat_dim = semantic_feat_dim
        self.use_semantic_features = use_semantic_features
        self.use_hod_motion_module = use_hod_motion_module
        self.hod_motion_feat_dim = hod_motion_feat_dim
        self.use_cross_motion_module = use_cross_motion_module
        self.cross_motion_feat_dim = cross_motion_feat_dim
        self.fusion_type = fusion_type
        self.final_fusion_type = final_fusion_type

        active_dims = []
        if use_semantic_features:
            active_dims.append(semantic_feat_dim)
        if use_hod_motion_module:
            active_dims.append(hod_motion_feat_dim)
        if use_cross_motion_module:
            active_dims.append(cross_motion_feat_dim)
        if not active_dims:
            raise ValueError("At least one fusion branch must be active.")

        if output_dim is not None:
            self.output_dim = output_dim
        elif fusion_type == "add":
            self.output_dim = active_dims[0]
        elif fusion_type in {"concat", "self_attention", "cross_attention", "self_cross_attention"}:
            self.output_dim = sum(active_dims) if final_fusion_type in {None, "concat"} else active_dims[0]
        elif fusion_type == "gated_softmax_fusion":
            self.output_dim = gated_fusion_output_dim
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")

        if fusion_type in {"self_attention", "self_cross_attention"}:
            self.self_attention_across_dim = self_attention_across_dim
            self.semantic_attn = (
                SelfAttention(
                    dim=semantic_feat_dim,
                    heads=self_attention_num_heads,
                    dropout_rate=self_attention_dropout_rate,
                    use_ffn=self_attention_use_ffn,
                )
                if use_semantic_features
                else None
            )
            self.hod_motion_attn = (
                SelfAttention(
                    dim=hod_motion_feat_dim,
                    heads=self_attention_num_heads,
                    dropout_rate=self_attention_dropout_rate,
                    use_ffn=self_attention_use_ffn,
                )
                if use_hod_motion_module
                else None
            )
            self.cross_motion_attn = (
                SelfAttention(
                    dim=cross_motion_feat_dim,
                    heads=self_attention_num_heads,
                    dropout_rate=self_attention_dropout_rate,
                    use_ffn=self_attention_use_ffn,
                )
                if use_cross_motion_module
                else None
            )

        if fusion_type in {"cross_attention", "self_cross_attention"}:
            self.cross_attention_along_dim = cross_attention_along_dim
            self.cross_attention_use_ffn = cross_attention_use_ffn
            self._init_cross_attention(
                semantic_feat_dim,
                hod_motion_feat_dim,
                cross_motion_feat_dim,
                cross_attention_num_heads,
                cross_attention_num_layers,
                cross_attention_use_ffn,
                cross_attention_along_dim,
            )

        if fusion_type == "gated_softmax_fusion":
            self._init_gated_softmax_fusion(
                active_dims,
                gated_fusion_hidden_dim,
                gated_fusion_output_dim,
            )

        logger.info(
            "Initialized FusionModule: type=%s, output_dim=%s, use_semantic=%s, "
            "use_hod_motion=%s, use_cross_motion=%s",
            fusion_type,
            self.output_dim,
            use_semantic_features,
            use_hod_motion_module,
            use_cross_motion_module,
        )

    def forward(self, semantic_feat, hod_motion_feat=None, cross_motion_feat=None):
        if self.fusion_type == "add":
            return self._add_fusion(semantic_feat, hod_motion_feat, cross_motion_feat)
        if self.fusion_type == "concat":
            return self._concat_fusion(semantic_feat, hod_motion_feat, cross_motion_feat)
        if self.fusion_type == "self_attention":
            return self._self_attention_fusion(semantic_feat, hod_motion_feat, cross_motion_feat)
        if self.fusion_type == "cross_attention":
            return self._cross_attention(semantic_feat, hod_motion_feat, cross_motion_feat)
        if self.fusion_type == "self_cross_attention":
            self_attended, names = self._self_attention_fusion(
                semantic_feat, hod_motion_feat, cross_motion_feat, return_per_modality=True
            )
            by_name = dict(zip(names, self_attended))
            return self._cross_attention(
                by_name.get("semantic"),
                by_name.get("hod_motion"),
                by_name.get("cross_motion"),
            )
        if self.fusion_type == "gated_softmax_fusion":
            return self._gated_softmax_fusion(semantic_feat, hod_motion_feat, cross_motion_feat)
        raise ValueError(f"Unsupported fusion type: {self.fusion_type}")

    def _active_features(self, semantic_feat, hod_motion_feat, cross_motion_feat):
        features = {}
        if self.use_semantic_features and semantic_feat is not None:
            features["semantic"] = semantic_feat
        if self.use_hod_motion_module and hod_motion_feat is not None:
            features["hod_motion"] = hod_motion_feat
        if self.use_cross_motion_module and cross_motion_feat is not None:
            features["cross_motion"] = cross_motion_feat
        if not features:
            raise ValueError("Fusion requires at least one active feature tensor.")
        return features

    def _reference_feat(self, semantic_feat, hod_motion_feat, cross_motion_feat):
        return next(iter(self._active_features(semantic_feat, hod_motion_feat, cross_motion_feat).values()))

    def _add_fusion(self, semantic_feat, hod_motion_feat=None, cross_motion_feat=None):
        parts = list(self._active_features(semantic_feat, hod_motion_feat, cross_motion_feat).values())
        fused = parts[0]
        for part in parts[1:]:
            fused = fused + part
        return fused

    def _concat_fusion(self, semantic_feat, hod_motion_feat=None, cross_motion_feat=None):
        return torch.cat(
            list(self._active_features(semantic_feat, hod_motion_feat, cross_motion_feat).values()),
            dim=-1,
        )

    def _apply_attention_along_dim(self, feat, attn, b, t, p):
        if self.self_attention_across_dim == "spatial":
            feat = rearrange(feat, "b t p d -> (b t) p d")
            feat = attn(feat)
            return rearrange(feat, "(b t) p d -> b t p d", b=b, t=t)
        if self.self_attention_across_dim == "temporal":
            feat = rearrange(feat, "b t p d -> (b p) t d")
            feat = attn(feat)
            return rearrange(feat, "(b p) t d -> b t p d", b=b, p=p)
        raise ValueError("self_attention_across_dim must be 'spatial' or 'temporal'")

    def _self_attention_fusion(
        self,
        semantic_feat,
        hod_motion_feat=None,
        cross_motion_feat=None,
        return_per_modality=False,
    ):
        ref = self._reference_feat(semantic_feat, hod_motion_feat, cross_motion_feat)
        b, t, p, _ = ref.shape
        features = {}
        if self.semantic_attn is not None and semantic_feat is not None:
            features["semantic"] = self._apply_attention_along_dim(
                semantic_feat, self.semantic_attn, b, t, p
            )
        if self.hod_motion_attn is not None and hod_motion_feat is not None:
            features["hod_motion"] = self._apply_attention_along_dim(
                hod_motion_feat, self.hod_motion_attn, b, t, p
            )
        if self.cross_motion_attn is not None and cross_motion_feat is not None:
            features["cross_motion"] = self._apply_attention_along_dim(
                cross_motion_feat, self.cross_motion_attn, b, t, p
            )
        if return_per_modality:
            return list(features.values()), list(features.keys())
        return torch.cat(list(features.values()), dim=-1)

    def _init_cross_attention(
        self,
        semantic_dim,
        hod_motion_dim,
        cross_motion_dim,
        num_heads,
        num_layers,
        use_ffn,
        along_dim,
    ):
        if along_dim not in {"spatial", "temporal", "both"}:
            raise ValueError("cross_attention_along_dim must be 'spatial', 'temporal', or 'both'")

        def make_transformer(dim):
            layer = CrossAttentionEncoderLayer(d_model=dim, nhead=num_heads, use_ffn=use_ffn)
            return CrossAttentionTransformerEncoder(layer, num_layers=num_layers)

        self.cross_transformers = nn.ModuleDict()
        dims = {}
        if self.use_semantic_features:
            dims["semantic"] = semantic_dim
        if self.use_hod_motion_module:
            dims["hod_motion"] = hod_motion_dim
        if self.use_cross_motion_module:
            dims["cross_motion"] = cross_motion_dim
        for name, dim in dims.items():
            axes = {}
            if along_dim in {"spatial", "both"}:
                axes["spatial"] = make_transformer(dim)
            if along_dim in {"temporal", "both"}:
                axes["temporal"] = make_transformer(dim)
            self.cross_transformers[name] = nn.ModuleDict(axes)

    def _run_cross_attention_axis(self, features, axis, b, t, p):
        if axis == "spatial":
            seq_features = {
                name: rearrange(feat, "b t p d -> (b t) p d")
                for name, feat in features.items()
            }
        else:
            seq_features = {
                name: rearrange(feat, "b t p d -> (b p) t d")
                for name, feat in features.items()
            }

        attended = {}
        for name, feat in seq_features.items():
            memory = torch.cat([other for other_name, other in seq_features.items() if other_name != name], dim=1)
            attended[name] = self.cross_transformers[name][axis](feat, memory)

        if axis == "spatial":
            return {
                name: rearrange(feat, "(b t) p d -> b t p d", b=b, t=t)
                for name, feat in attended.items()
            }
        return {
            name: rearrange(feat, "(b p) t d -> b t p d", b=b, p=p)
            for name, feat in attended.items()
        }

    def _cross_attention(self, semantic_feat, hod_motion_feat=None, cross_motion_feat=None):
        ref = self._reference_feat(semantic_feat, hod_motion_feat, cross_motion_feat)
        b, t, p, _ = ref.shape
        features = self._active_features(semantic_feat, hod_motion_feat, cross_motion_feat)
        if len(features) < 2:
            raise ValueError("Cross-attention fusion requires at least two active modalities.")
        axes = ("spatial", "temporal") if self.cross_attention_along_dim == "both" else (self.cross_attention_along_dim,)
        for axis in axes:
            features = self._run_cross_attention_axis(features, axis, b, t, p)
        return torch.cat(list(features.values()), dim=-1)

    def _init_gated_softmax_fusion(self, active_dims, hidden_dim, output_dim):
        self.gated_value_projs = nn.ModuleList([nn.Linear(dim, output_dim) for dim in active_dims])
        self.gated_gate = nn.Sequential(
            nn.LayerNorm(sum(active_dims)),
            nn.Linear(sum(active_dims), hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, len(active_dims)),
        )
        self.gated_softmax = nn.Softmax(dim=-1)

    def _gated_softmax_fusion(self, semantic_feat, hod_motion_feat=None, cross_motion_feat=None):
        features = list(self._active_features(semantic_feat, hod_motion_feat, cross_motion_feat).values())
        gate_logits = self.gated_gate(torch.cat(features, dim=-1))
        weights = self.gated_softmax(gate_logits)
        projected = [proj(feat) for proj, feat in zip(self.gated_value_projs, features)]
        fused = sum(weights[..., i : i + 1] * projected[i] for i in range(len(projected)))
        return fused
