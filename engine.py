# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils

from torchvision.utils import save_image

from datasets.demo import denorm
import shutil
import pandas as pd

# ----------------------------------------------------------------------------------
def train_one_epoch_st(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,lr_scheduler,
                    device: torch.device,logger, epoch: int, save_path:str,args,max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    for it,(samples,style_images, targets) in metric_logger.log_every(data_loader, print_freq,logger, header):
#         
        samples = samples.to(device)
        style_images = style_images.to(device)
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples,style_images)
                
    
        loss_dict = criterion(outputs, samples,style_images)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            logger.info("Loss is {}, stopping training".format(loss_value))
            logger.info(loss_dict_reduced)
            sys.exit(1)
        if it % 2000==0:
            if not os.path.exists(os.path.join(save_path,"train_outputs")):
                os.makedirs(os.path.join(save_path,"train_outputs"))
            if not os.path.exists(os.path.join(save_path,"train_content_images")):
                os.makedirs(os.path.join(save_path,"train_content_images"))
            if not os.path.exists(os.path.join(save_path,"train_style_images")):
                os.makedirs(os.path.join(save_path,"train_style_images"))
                          
            if isinstance(outputs, tuple):
                outputs,_=outputs
            outputs=denorm(outputs, device)
            samples.tensors=denorm(samples.tensors, device)
            style_images.tensors=denorm(style_images.tensors, device)
            save_image(outputs,os.path.join(save_path,"train_outputs",f'{epoch:04}_{it:08}.png')  )   
            save_image(samples.tensors,os.path.join(save_path,"train_content_images",f'{epoch:04}_{it:08}.png' )  )  
            save_image(style_images.tensors,os.path.join(save_path,"train_style_images",f'{epoch:04}_{it:08}.png' )  )     # .clamp(0,1)
            
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
#         metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if it % 4000==0:
            if not os.path.exists(os.path.join(save_path,"checkpoint")):
                os.makedirs(os.path.join(save_path,"checkpoint"))
            checkpoint_path = os.path.join(save_path,"checkpoint",f'checkpoint{epoch:04}.pth')

            utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
            }, checkpoint_path)

        torch.cuda.empty_cache()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info("Averaged stats:{}".format(metric_logger))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate_st(model, criterion, postprocessors, data_loader, base_ds, device, logger,epoch,save_path):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    for it,(samples,style_images, targets) in metric_logger.log_every(data_loader, 100,logger, header):

        samples = samples.to(device)
        style_images = style_images.to(device)
        
        outputs = model(samples,style_images)  
        
        
        loss_dict = criterion(outputs, samples,style_images)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if it % 200==0:
            if not os.path.exists(os.path.join(save_path,"eval_outputs")):
                os.makedirs(os.path.join(save_path,"eval_outputs"))
            if not os.path.exists(os.path.join(save_path,"eval_content_images")):
                os.makedirs(os.path.join(save_path,"eval_content_images"))
            if not os.path.exists(os.path.join(save_path,"eval_style_images")):
                os.makedirs(os.path.join(save_path,"eval_style_images"))
                
          
            if isinstance(outputs, tuple):
                outputs,_=outputs
            outputs=denorm(outputs, device)
            samples.tensors=denorm(samples.tensors, device)
            style_images.tensors=denorm(style_images.tensors, device)
            
            save_image(outputs,os.path.join(save_path,"eval_outputs",f'{epoch:04}_{it:08}.png' )  )  
            save_image(samples.tensors,os.path.join(save_path,"eval_content_images",f'{epoch:04}_{it:08}.png' )  )  
            save_image(style_images.tensors,os.path.join(save_path,"eval_style_images",f'{epoch:04}_{it:08}.png' )  )     # .clamp(0,1)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    return stats

def save_batch_images(save_path,outputs,samples,style_images,epoch,it,device):
    
    outputs=denorm(outputs, device)
    samples.tensors=denorm(samples.tensors, device)
    style_images.tensors=denorm(style_images.tensors, device)
#             print("outputs.shape:",outputs.shape)
    save_image(outputs,os.path.join(save_path,"test_outputs",f'{epoch:04}',f'{epoch:04}_{it:08}.png' )  )  
    save_image(samples.tensors,os.path.join(save_path,"test_content_images",f'{epoch:04}',f'{epoch:04}_{it:08}.png' )  )  
    save_image(style_images.tensors,os.path.join(save_path,"test_style_images",f'{epoch:04}',f'{epoch:04}_{it:08}.png' )  ) 

def test_st(model, criterion, postprocessors, data_loader,  device, logger,epoch,save_path):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    print("len(data_loader):",len(data_loader))
    if os.path.exists(os.path.join(save_path,"test_outputs",f'{epoch:04}')):
        shutil.rmtree(os.path.join(save_path,"test_outputs",f'{epoch:04}'))
    os.makedirs(os.path.join(save_path,"test_outputs",f'{epoch:04}'))
    tmp_out=[]
    for it,(samples,style_images, targets) in metric_logger.log_every(data_loader, 100,logger, header):
#         if it<=3:
#             continue
        samples = samples.to(device)
        style_images = style_images.to(device)
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples,style_images)  
        
    
        loss_dict = criterion(outputs, samples,style_images)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if it % 1==0:
            if not os.path.exists(os.path.join(save_path,"test_content_images",f'{epoch:04}')):
                os.makedirs(os.path.join(save_path,"test_content_images",f'{epoch:04}'))
            if not os.path.exists(os.path.join(save_path,"test_style_images",f'{epoch:04}')):
                os.makedirs(os.path.join(save_path,"test_style_images",f'{epoch:04}'))
                
          
            if isinstance(outputs, tuple):
                outputs,_=outputs
                
#             if False:
            if "content_image_name" in targets[0]:
                for i in range(len(outputs)):
                    c_name=targets[i]["content_image_name"]
                    s_name=targets[i]["style_image_name"]
                    save_name="{}_{}".format(c_name,s_name)
                    
                    output_i=denorm(outputs[i], device)
                    save_image(output_i,os.path.join(save_path,"test_outputs",f'{epoch:04}',f'{epoch:04}_{save_name}.png' )  )  
                    
                    sample_i=denorm(samples.tensors[i], device)
                    save_image(sample_i,os.path.join(save_path,"test_content_images",f'{epoch:04}',f'{epoch:04}_{save_name}.png' )  )  
                    
            else:
                save_batch_images(save_path,outputs,samples,style_images,epoch,it,device)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    return stats
