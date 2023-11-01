# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import torch
from .models import build_ACT_model, build_CNNMLP_model
import click


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float) # will be overridden
    parser.add_argument('--lr_backbone', default=1e-5, type=float) # will be overridden
    parser.add_argument('--batch_size', default=2, type=int) # not used
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int) # not used
    parser.add_argument('--lr_drop', default=200, type=int) # not used
    parser.add_argument('--clip_max_norm', default=0.1, type=float, # not used
                        help='gradient clipping max norm')

    # Model parameters
    # * Backbone
    parser.add_argument('--backbone', default='resnet18', type=str, # will be overridden
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--camera_names', default=[], type=list, # will be overridden
                        help="A list of camera names")

    # * Transformer
    parser.add_argument('--enc_layers', default=4, type=int, # will be overridden
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int, # will be overridden
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int, # will be overridden
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int, # will be overridden
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int, # will be overridden
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=400, type=int, # will be overridden
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # repeat args in train just to avoid error. Will not be used
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--dataset_dir', action='store', type=str, help='dataset_dir', required=False)
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=False)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=False)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=False)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=False)
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--temporal_agg', action='store_true')

    parser.add_argument('-p', '--rollout_path', type=str, help='absolute path of the rollout', default=None)
    parser.add_argument('-f', '--rollout_format', type=click.Choice(['RoboHive', 'RoboSet']), help='Data format', default='RoboHive')
    parser.add_argument('-m', '--mode', type=click.Choice(['record', 'render', 'playback', 'recover','policy']), help='How to examine rollout', default='playback')
    parser.add_argument('-hor', '--horizon', type=int, help='Rollout horizon, when mode is record', default=-1)
    parser.add_argument('-num_repeat', '--num_repeat', type=int, help='number of repeats for the rollouts', default=1)
    parser.add_argument('-r', '--render', type=click.Choice(['onscreen', 'offscreen', 'none']), help='visualize onscreen or offscreen', default='onscreen')
    parser.add_argument('-c', '--camera_name', type=str, default=[None,], help=('list of camera names for rendering'))
    parser.add_argument('-fs', '--frame_size', type=tuple, default=(424, 240), help=('Camera frame size for rendering'))
    parser.add_argument('-o', '--output_dir', type=str, default='/checkpoint/homanga/cactiv2/robohivelogs', help=('Directory to save the outputs'))
    parser.add_argument('-on', '--output_name', type=str, default=None, help=('The name to save the outputs as'))
    parser.add_argument('-sp', '--save_paths', type=bool, default=False, help=('Save the rollout paths'))
    parser.add_argument('-cp', '--compress_paths', type=bool, default=True, help=('compress paths. Remove obs and env_info/state keys'))
    parser.add_argument('-pp', '--plot_paths', type=bool, default=False, help=('2D-plot of individual paths'))
    parser.add_argument('-ea', '--env_args', type=str, default=None, help=('env args. E.g. --env_args "{\'is_hardware\':True}"'))
    parser.add_argument('-ns', '--noise_scale', type=float, default=0.0, help=('Noise amplitude in randians}"'))
    parser.add_argument('-run_name', '--run_name', action='store', type=str, help='run name for logs', required=False)
    parser.add_argument('--mask_id', type=int, default=0, help=('mask id'))

    parser.add_argument('--num_episodes', default=500, action='store', type=int, help='total episodes in folder', required=False)

    # add this for multi-task embedding condition
    parser.add_argument('--multi_task', action='store_true')

    return parser


def build_ACT_model_and_optimizer(args_override):
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    for k, v in args_override.items():
        setattr(args, k, v)

    model = build_ACT_model(args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return model, optimizer


def build_CNNMLP_model_and_optimizer(args_override):
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    for k, v in args_override.items():
        setattr(args, k, v)

    model = build_CNNMLP_model(args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return model, optimizer

