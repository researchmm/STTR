# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding,build_position_encoding_ours


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, backbone_layer,train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
#             return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            return_layer_idxs=list(range(1,backbone_layer+1))
            return_layers = {}
            for i,rly in enumerate(return_layer_idxs):
                return_layers["layer{}".format(rly)]=str(i)
        else:
#             return_layers = {'layer4': "0"}
            return_layers = {"layer{}".format(backbone_layer): "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        
        layer2dim_dict={
            1: 256, 2: 512, 3: 1024, 4: 2048
        }
        layer2reduce_dict={ # 2**
            1: 2, 
            2: 3, 
            3: 4, 
            4: 5
        }
        self.num_channels =layer2dim_dict[backbone_layer]
        self.reduce_times=layer2reduce_dict[backbone_layer]
#         self.num_channels = num_channels
#         self.num_channels = 512
        

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone_50(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str, backbone_layer,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone,backbone_layer, train_backbone, num_channels, return_interm_layers)
        
        
        

class ResBlock(nn.Module):
    """
    Usual full pre-activation ResNet bottleneck block.
    For more information see
    He, K., Zhang, X., Ren, S., & Sun, J. (2016, October).
    Identity mappings in deep residual networks.
    European Conference on Computer Vision (pp. 630-645).
    Springer, Cham.
    ArXiv link: https://arxiv.org/abs/1603.05027
    """
    def __init__(self, outer_dim, inner_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(outer_dim),
            nn.LeakyReLU(),
            nn.Conv2d(outer_dim, inner_dim, 1),
            nn.BatchNorm2d(inner_dim),
            nn.LeakyReLU(),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.BatchNorm2d(inner_dim),
            nn.LeakyReLU(),
            nn.Conv2d(inner_dim, outer_dim, 1),
        )

    def forward(self, input):
        return input + self.net(input)

class ResBlock_nonorm(nn.Module):
    """
    Usual full pre-activation ResNet bottleneck block.
    For more information see
    He, K., Zhang, X., Ren, S., & Sun, J. (2016, October).
    Identity mappings in deep residual networks.
    European Conference on Computer Vision (pp. 630-645).
    Springer, Cham.
    ArXiv link: https://arxiv.org/abs/1603.05027
    """
    def __init__(self, outer_dim, inner_dim):
        super().__init__()
        self.net = nn.Sequential(
#             nn.BatchNorm2d(outer_dim),
#             nn.LeakyReLU(),
            nn.Conv2d(outer_dim, inner_dim, 1),
#             nn.BatchNorm2d(inner_dim),
            nn.LeakyReLU(),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
#             nn.BatchNorm2d(inner_dim),
            nn.LeakyReLU(),
            nn.Conv2d(inner_dim, outer_dim, 1),
        )

    def forward(self, input):
        return input + self.net(input)
    
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=True)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=True)

class Backbone_simple(nn.Module):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self):
#         backbone = getattr(torchvision.models, name)(
#             replace_stride_with_dilation=[False, False, dilation],
#             pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
#         num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__()
#         self.hidden_dim=hidden_dim
#         self.body = nn.Sequential(
#                 nn.Conv2d(3, 128, kernel_size=3,padding=1),
# #                 nn.BatchNorm2d(128),
#                 nn.LeakyReLU(),
#                 nn.Conv2d(128, hidden_dim, kernel_size=3,padding=1),
            
# #                 ResBlock(outer_dim=128, inner_dim=128),
# #                 nn.Conv2d(128, 256, kernel_size=1)
#             )
#         self.body = nn.Sequential(
#                 nn.Conv2d(3, 64, kernel_size=4,padding=1,stride=2),
#                 nn.BatchNorm2d(64),
#                 nn.LeakyReLU(),
#                 nn.Conv2d(64, 128, kernel_size=4,padding=1,stride=2),
#                 nn.BatchNorm2d(128),
#                 nn.LeakyReLU(),
#                 nn.Conv2d(128, 256, kernel_size=4,padding=1,stride=2),
#             )
        self.body = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3,padding=1),
                ResBlock(outer_dim=64, inner_dim=32),
                ResBlock(outer_dim=64, inner_dim=32)
            )
        self.num_channels=64
    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        xs ={"0": xs}
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out
        
class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone_50(args,backbone_layer=4):
#     position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone_50(args.backbone,backbone_layer,train_backbone, return_interm_layers, args.dilation)
#     backbone = Backbone()
#     model = Joiner(backbone, position_embedding)
#     model.num_channels = backbone.num_channels
    return backbone



def build_backbone_simple(args):
#     train_backbone = args.lr_backbone > 0
#     return_interm_layers = args.masks
#     backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    backbone = Backbone_simple()
#     model = Joiner(backbone, position_embedding)
#     model.num_channels = backbone.num_channels
    return backbone
