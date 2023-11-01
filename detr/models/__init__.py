# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

## UNCOMMENT FOR SINGLE TASK
# from .detr_vae_single_task import build as build_vae
# from .detr_vae_single_task import build_cnnmlp as build_cnnmlp



# ## UNCOMMENT FOR MULTI-TASK 
from .detr_vae import build as build_vae
from .detr_vae import build_cnnmlp as build_cnnmlp


def build_ACT_model(args):
    return build_vae(args)

def build_CNNMLP_model(args):
    return build_cnnmlp(args)

