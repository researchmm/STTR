# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision



def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    from .demo import build as build_coco_wikiart
    return build_coco_wikiart(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
    
def build_video_dataset(image_set, args,video_frames_path,style_path):
    from .general_wikiart_nofold_video import build as build_coco_wikiart
    return build_coco_wikiart(image_set, args,video_frames_path,style_path)