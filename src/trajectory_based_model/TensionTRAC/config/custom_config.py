#!/usr/bin/env python3

"""Add custom configs and default values"""

from fvcore.common.config import CfgNode

# pylint: disable=line-too-long


def add_custom_config(cfg):
    """Add custom configs."""
    cfg.DATA.PATH_TO_TENSIONTRAC_PT_DATA = "tension_laparoscopic_surgery/data/TRAC_features/2d_intra_cross_sem"

    # wandb config
    cfg.WANDB = CfgNode()
    cfg.WANDB.PROJECT = "TensionTRAC"
    cfg.WANDB.ENTITY = ""
    cfg.WANDB.ID = ""
    cfg.WANDB.EXP_NAME = ""

    # few shot config
    cfg.FEW_SHOT = CfgNode()
    cfg.FEW_SHOT.N_WAY = 5
    cfg.FEW_SHOT.K_SHOT = 1
    cfg.FEW_SHOT.TRAIN_QUERY_PER_CLASS = 6
    cfg.FEW_SHOT.TEST_QUERY_PER_CLASS = 1
    cfg.FEW_SHOT.TRAIN_EPISODES = 100000
    cfg.FEW_SHOT.TEST_EPISODES = 10000
    cfg.FEW_SHOT.PATCH_TOKENS_AGG = "spatial"
    cfg.FEW_SHOT.USE_MODEL = True
    cfg.FEW_SHOT.DIST_NORM = "none"
    cfg.FEW_SHOT.TRAIN_OG_EPISODES = False
    cfg.FEW_SHOT.CLASS_LOSS_LAMBDA = 1.0
    cfg.FEW_SHOT.Q2S_LOSS_LAMBDA = 1.0
    # =========================================================================================
    # few-shot classifier:
    # - frame_proto: original prototype frame-token matching
    # - traj_instance: trajectory-native instance matching over support instances
    # - dual: frame_proto + TRAJ_SCORE_LAMBDA * traj_instance
    cfg.FEW_SHOT.CLASSIFIER_MODE = "dual"
    cfg.FEW_SHOT.TRAJ_SCORE_LAMBDA = 1.0
    cfg.FEW_SHOT.TRAJ_TOPK = CfgNode()
    cfg.FEW_SHOT.TRAJ_TOPK.ENABLE = True
    cfg.FEW_SHOT.TRAJ_TOPK.RATIO = 0.5
    # "dual_only" (default), "traj_only", or "all"
    cfg.FEW_SHOT.TRAJ_TOPK.APPLY_TO = "dual_only"
    cfg.FEW_SHOT.TRAJ_REWEIGHT = CfgNode()
    cfg.FEW_SHOT.TRAJ_REWEIGHT.ENABLE = True
    # choose cls source for trajectory-similarity reweight:
    # - video: pointformer video cls token
    # - dino: DINOv2 video cls token
    # - fusion: weighted fusion of video and DINO cls token
    cfg.FEW_SHOT.TRAJ_REWEIGHT.SIM_CLS_SOURCE = "fusion"
    cfg.FEW_SHOT.TRAJ_REWEIGHT.FUSION_ALPHA = 0.6
    cfg.FEW_SHOT.TRAJ_REWEIGHT.TEMPERATURE = 0.5
    cfg.FEW_SHOT.TRAJ_REWEIGHT.THRESHOLD = 0.0
    cfg.FEW_SHOT.TRAJ_REWEIGHT.MIN_WEIGHT = 0.1
    cfg.FEW_SHOT.TRAJ_REWEIGHT.USE_VISIBILITY_MASK = True
    # =========================================================================================

    # point info config
    cfg.POINT_INFO = CfgNode()
    cfg.POINT_INFO.ENABLE = True
    cfg.POINT_INFO.GRID_SIZE = 16
    cfg.POINT_INFO.NAME = ""
    cfg.POINT_INFO.NUM_POINTS_TO_SAMPLE = 256
    cfg.POINT_INFO.SAMPLING_TYPE = "random"
    cfg.POINT_INFO.POINT_DIM = 2
    cfg.POINT_INFO.HOD_ORIENTATION_HIST_TYPE = "2d"
    cfg.POINT_INFO.PT_FIX_SAMPLING_TRAIN = False
    cfg.POINT_INFO.PT_FIX_SAMPLING_TEST = False
    cfg.POINT_INFO.USE_PT_QUERY_MASK = False
    cfg.POINT_INFO.OBJ_ID_KEY = "obj_ids"
    cfg.POINT_INFO.HOD = CfgNode()
    cfg.POINT_INFO.HOD.NUM_BINS = 32
    cfg.POINT_INFO.HOD.NUM_CLUSTERS = 16
    cfg.POINT_INFO.HOD_MIN = True
    cfg.POINT_INFO.HOD.GET_FEAT = True
    cfg.POINT_INFO.HOD.TEMPORAL_PYRAMID = False
    cfg.POINT_INFO.HOD.TEMPORAL_PYRAMID_LEVELS = 3
    cfg.POINT_INFO.HOD.PRESERVE_TEMPORAL = True
    cfg.POINT_INFO.USE_CORRELATION = False

    # motion module config
    cfg.MODEL.MOTION_MODULE = CfgNode()
    cfg.MODEL.MOTION_MODULE.USE_HOD_MOTION_MODULE = False
    cfg.MODEL.MOTION_MODULE.USE_CROSS_MOTION_MODULE = False
    cfg.MODEL.MOTION_MODULE.INNER_MODULE_TYPE = "hod"
    cfg.MODEL.MOTION_MODULE.INTER_MODULE_TYPE = "legacy_cross"
    cfg.MODEL.MOTION_MODULE.INNER_HIDDEN_DIM = 128
    cfg.MODEL.MOTION_MODULE.INTER_HIDDEN_DIM = 128
    cfg.MODEL.MOTION_MODULE.INNER_KERNEL_SIZE = 3
    cfg.MODEL.MOTION_MODULE.K_NEIGHBORS = 8
    cfg.MODEL.MOTION_MODULE.DROPOUT = 0.1
    cfg.MODEL.MOTION_MODULE.SMOOTH_Z = True
    # 0 = use embed_dim (768); e.g. 384 for compressed motion output
    cfg.MODEL.MOTION_MODULE.OUT_FEATURE_DIM = 0
    cfg.MODEL.APPEARANCE_MODULE_DISABLE = False

    cfg.MODEL.FUSION = CfgNode()
    cfg.MODEL.FUSION.TYPE = "add"
    cfg.MODEL.FUSION.OUTPUT_DIM = None
    cfg.MODEL.FUSION.SELF_ATTENTION_NUM_HEADS = 4
    cfg.MODEL.FUSION.SELF_ATTENTION_DROPOUT_RATE = 0.0
    cfg.MODEL.FUSION.SELF_ATTENTION_USE_FFN = False
    cfg.MODEL.FUSION.SELF_ATTENTION_USE_POS_ENCODING = False
    cfg.MODEL.FUSION.SELF_ATTENTION_POS_ENCODING_TYPE = "fixed"
    cfg.MODEL.FUSION.SELF_ATTENTION_ACROSS_DIM = "spatial"
    cfg.MODEL.FUSION.SELF_ATTENTION_POS_ENCODING_ACROSS_DIM = "same"
    cfg.MODEL.FUSION.CROSS_ATTENTION_NUM_HEADS = 4
    cfg.MODEL.FUSION.CROSS_ATTENTION_NUM_LAYERS = 2
    cfg.MODEL.FUSION.CROSS_ATTENTION_USE_FFN = False
    cfg.MODEL.FUSION.CROSS_ATTENTION_ALONG_DIM = "spatial"
    cfg.MODEL.FUSION.GATED_FUSION_HIDDEN_DIM = 128
    cfg.MODEL.FUSION.GATED_FUSION_OUTPUT_DIM = 768
    cfg.MODEL.FUSION.FINAL_FUSION_TYPE = "concat"
    cfg.MODEL.FUSION.SELF_CROSS_ATTENTION_USE_FFN = False
    cfg.MODEL.FUSION.USE_SEMANTIC_FEATURES = True
    # 0 = use dino_feat_size (768); e.g. 600 before fusion self-attention
    cfg.MODEL.FUSION.SEMANTIC_FUSION_DIM = 0

    return cfg
