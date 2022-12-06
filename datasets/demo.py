# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

import datasets.transforms as T

from torchvision import transforms as Tr

import glob
from PIL import Image
from PIL import ImageFile
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os


std = [0.229, 0.224, 0.225]
mean = [0.485, 0.456, 0.406]

def denorm(tensor, device):
    std_ = torch.Tensor(std).reshape(-1, 1, 1).to(device)
    mean_ = torch.Tensor(mean).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std_ + mean_, 0, 1)
    return res

class CocoStyleTransfer(torchvision.datasets.CocoDetection):
    def __init__(self,content_folder, style_folder,img_size):
#         super(CocoStyleTransfer, self).__init__(coco_img_folder, coco_ann_file)
        
        style_images = glob.glob(str( style_folder) + '/*')
#         self._transforms = transforms
        
        self.std = std
        self.mean = mean
        self._transforms = Tr.Compose([
            Tr.ToTensor(),
            Tr.Normalize(self.mean, self.std)
        ])
        
        content_images =glob.glob(str(content_folder) + '/*')
        print("len(content_images),len(style_images):",len(content_images),len(style_images))
        self.images_pairs = [[x,y] for x in content_images for y in style_images ] 
        self.img_size=img_size
    def center_crop(self,img,max_img_size=600):
        width, height = img.size   # Get dimensions
        if width>=max_img_size or height>=max_img_size:
            new_width=max_img_size
            new_height=int(max_img_size/width*height)
            left = (width - new_width)/2
            top = (height - new_height)/2
            right = (width + new_width)/2
            bottom = (height + new_height)/2

            img = img.crop((left, top, right, bottom))
        return img
    
    def img_resize(self,im,div=32):
        desired_size=self.img_size
        h, w = im.size
        if h>w:
            new_h=desired_size
            new_w=int(new_h/h*w)
        else:
            new_w=desired_size
            new_h=int(new_w/w*h)
            
        
        new_w = (new_w%div==0) and new_w or (new_w + (div-(new_w%div)))
        new_h = (new_h%div==0) and new_h or (new_h + (div-(new_h%div)))
            
        new_im  = im.resize((new_h,new_w), Image.ANTIALIAS).convert("RGB")
        
        noise_new_im=np.array(new_im)
#         sigma=3
#         noise_new_im+=+np.random.randn(new_w,new_h,3) * sigma / 255
        return noise_new_im
        
    def image2div(self,img,div=32):
        width, height = img.size
        nw = (width%div==0) and width or (width + (div-(width%div)))
        nh = (height%div==0) and height or (height + (div-(height%div)))

        img = img.resize( (nw,nh), Image.ANTIALIAS)
        
        return img
        
    def __getitem__(self, pair_idx):
        content_image_path, style_image_path = self.images_pairs[pair_idx]
        c_name=os.path.basename(content_image_path).split(".")[0]
        s_name=os.path.basename(style_image_path).split(".")[0]
        target = {'content_image_name': c_name, 'style_image_name': s_name}
        img = Image.open(content_image_path).convert("RGB")
        style_image = Image.open(style_image_path).convert("RGB")
        style_image = self.img_resize(style_image)
        img = self.img_resize(img)
        
        if self._transforms is not None:
            style_image = self._transforms(style_image)
            img = self._transforms(img)
        return img, style_image, target
    
    def __len__(self):
        return len(self.images_pairs)

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target

def make_coco_wikiart_transforms(image_set):
    
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return normalize

def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    content_folder = Path(args.in_content_folder)
    style_folder = Path(args.style_folder)
    assert content_folder.exists(), f'provided COCO path {content_folder} does not exist'
    assert style_folder.exists(), f'provided wikiart path {style_folder} does not exist'

#     wikiart_img_folder = WIKIART_PATH[image_set]
    dataset = CocoStyleTransfer(content_folder, style_folder,args.img_size)
    return dataset
