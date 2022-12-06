# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate_st, train_one_epoch_st,test_st
from models_istt import build_model
import logging
import pprint
import os


def get_args_parser_main():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    
#     parser.add_argument('--in_video_folders', default="", type=str)
    parser.add_argument('--in_content_folder', default="", type=str)
    parser.add_argument('--style_folder',default="", type=str)
    parser.add_argument('--output_dir', default="output_nofold_fix512",
                        help='path where to save, empty for no saving')
    
    parser.add_argument('--img_size', default=256, type=int)
    return parser
    

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    
#     parser.add_argument('--in_video_folders', default="", type=str)
    parser.add_argument('--in_content_folder', default="", type=str)
    parser.add_argument('--style_folder',default="", type=str)
    
    
    
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    #################### jianbo
    parser.add_argument('--model_type', default='nofold', type=str,
                        help="type of model")
    parser.add_argument('--fold_k', default=8, type=int,
                        help="Size of fold kernels")
    parser.add_argument('--fold_stride', default=6, type=int,
                        help="Size of fold kernels")
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--enorm', action='store_true')
    parser.add_argument('--dnorm', action='store_true')
    parser.add_argument('--tnorm', action='store_true')
    parser.add_argument('--model_pre', action='store_true')
    parser.add_argument('--cbackbone_layer', type=int, default=4,
                        help="")
    parser.add_argument('--sbackbone_layer', type=int, default=4,
                        help="")
    

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    
    parser.add_argument('--content_loss_coef', default=1.0, type=float)
    parser.add_argument('--style_loss_coef', default=1e4, type=float)
    parser.add_argument('--tv_loss_coef', default=0, type=float)
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--wikiart_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default="output_nofold_fix512",
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

def main(args,video_frames_path,styles_path):
    utils.init_distributed_mode(args)
    
    output_dir = Path(args.output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
        
    logging.basicConfig(filename=os.path.join(output_dir,"log_eval.txt"),
                    format='%(asctime)-15s %(message)s')
        
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    logger.info(pprint.pformat(args))
    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    
    model.to(device)
    
    logger.info(pprint.pformat(model))
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
        
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print('number of params:', n_parameters)
    logger.info('number of params: {}'.format(n_parameters))

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_val = build_dataset('val', args)

    print(video_frames_path,styles_path)
    print(len(dataset_val ))
    
    
    if args.distributed:
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)


    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn_st, num_workers=args.num_workers)

    checkpoint = torch.load(args.resume, map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'],strict=False)

    test_stats = test_st(
            model, criterion, postprocessors, data_loader_val, device,logger,checkpoint['epoch'], str(output_dir)
        )


        
        
if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    
    
    args = parser.parse_args([
                          "--batch_size","1",
                          "--style_loss_coef","10",
                          "--fold_k","5",
                          "--lr_backbone","1e-5",
                          "--lr","1e-5",
                          "--enc_layers","6",
                          "--dec_layers","6",
                          "--model_type","nofold",
                          "--enorm","--dnorm","--tnorm",
                          "--cbackbone_layer","2","--sbackbone_layer","4",
                          
                          "--dataset_file","demo",
                          "--resume","checkpoint_model/checkpoint0005.pth"  ,
                          "--img_size","512",
                          "--in_content_folder","inputs/content",
                          "--style_folder","inputs/style",
                          "--output_dir","outputs",
                             ])
    
    main(args,args.in_content_folder,args.style_folder)
    
    max_memory=torch.cuda.max_memory_allocated()
    msg='max_mem: {})'.format(max_memory)
    print(msg)