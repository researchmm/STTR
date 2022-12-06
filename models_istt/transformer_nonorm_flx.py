# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False,enorm=False,dnorm=False):
        super().__init__()

        
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before,enorm)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before and enorm else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before,dnorm)
        decoder_norm = nn.LayerNorm(d_model) if dnorm else None
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        
        ########### fold options
        self.fold_k=5
        self.fold_stride=self.fold_k-int(self.fold_k/4*3)
        self.fold_p=0
        

    def unfold_ours(self,tensor):
        # input: B,C,in_h,in_w
        # output: B,C*k*k,out_h*out_w
        B,C,in_h,in_w=tensor.shape
        out_h = (in_h - self.fold_k + 2 * self.fold_p) // self.fold_stride + 1
        out_w = (in_w - self.fold_k + 2 * self.fold_p) // self.fold_stride + 1
#         print("tensor.shape",tensor.shape)
        tensor = F.unfold(tensor, kernel_size=(self.fold_k, self.fold_k), padding=self.fold_p,stride=self.fold_stride)  #[B,C*k*k,out_h*out_w]
#         print("tensor.shape:",tensor.shape,B,C,self.fold_k,self.fold_k,out_h,out_w)
        tensor = tensor.reshape(B,C,self.fold_k,self.fold_k,out_h,out_w).permute(0,1,4,5,2,3).reshape(B,C*out_h*out_w,self.fold_k*self.fold_k)
        return tensor 
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

#     def forward(self, style_src, mask, src, query_pos_embed,pos_embed):
#         # flatten NxCxHxW to HWxNxC
# #         bs, c, h, w = src.shape
# #         print("0 src:",src.shape)
# #         src=F.unfold(src,kernel_size=(self.fold_k, self.fold_k), padding=self.fold_p,stride=self.fold_stride) 

# #         src=self.unfold_ours(src)
#         # [B,C*k*k,L]  L=[(H − k + 2P )/S+1] * [(W − k + 2P )/S+1]
# #         src=src.permute(2,0,1)   #[k*k,B,c*L] [k*k,B,C*out_h*out_w]
#         src = src.flatten(2).permute(2, 0, 1) # [320, 2, 256])  [H/32*W/32,B,C] 
# #         print("1 src:",src.shape)
        
# #         bs, c, h, w = style_src.shape
# #         print("0 style_src:",style_src.shape)
        
# #         style_src=F.unfold(style_src,kernel_size=(self.fold_k, self.fold_k), padding=self.fold_p,stride=self.fold_stride) 
#         # [B,C*k*k,L]  L=[(H − k + 2P )/S+1] * [(W − k + 2P )/S+1]
    
# #         style_src=self.unfold_ours(style_src)
# #         style_src=style_src.permute(2,0,1)   #[H*W,B,c*kh*kw]
#         style_src = style_src.flatten(2).permute(2, 0, 1)
# #         print("1 style_src:",style_src.shape)
# #         print("0 pos_embed:",pos_embed.shape)
#         pos_embed = pos_embed.flatten(2).permute(2, 0, 1) #[H*W,B,C]
# #         pos_embed = pos_embed.repeat(1, 1,self.K*self.K)       #[H*W,B,c*kh*kw]
# #         print("1 pos_embed:",pos_embed.shape)
# #         print("query_embed:",query_embed.shape)
#         query_pos_embed = query_pos_embed.flatten(2).permute(2, 0, 1) #[H*W,B,C]
#         mask = mask.flatten(1)
# #         print("mask:",mask.shape)
# #         tgt = torch.zeros_like(query_embed)
#         tgt = src
#         memory = self.encoder(style_src, src_key_padding_mask=mask, pos=pos_embed)
#         hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
#                           pos=pos_embed, query_pos=query_pos_embed)
# #         print("hs:",hs.shape) #torch.Size([layer_num, h*w, B, C])  
#         return hs.transpose(1, 2), memory.permute(1, 2, 0)

    def forward(self, style_src, mask, src, query_pos_embed,pos_embed):
#         print("style_src.shape, mask.shape, src.shape, query_pos_embed.shape,pos_embed.shape:")
#         print(style_src.shape, mask.shape, src.shape, query_pos_embed.shape,pos_embed.shape)
        
#         style_src.shape, mask.shape, src.shape, query_pos_embed.shape,pos_embed.shape:
#         torch.Size([1, 256, 16, 16]) torch.Size([1, 16, 16]) torch.Size([1, 256, 64, 48]) torch.Size([1, 256, 64, 48]) torch.Size([1, 256, 16, 16])

        
        # flatten NxCxHW to HWxNxC
#         if len(src.shape)==4:
#             bs, c, h, w = src.shape
#         src=F.unfold(src,kernel_size=(self.fold_k, self.fold_k), padding=self.fold_p,stride=self.fold_stride) 

#         src=self.unfold_ours(src)
        # [B,C*k*k,L]  L=[(H − k + 2P )/S+1] * [(W − k + 2P )/S+1]
#         src=src.permute(2,0,1)   #[k*k,B,c*L] [k*k,B,C*out_h*out_w]
        src = src.flatten(2).permute(2, 0, 1) # [320, 2, 256])  [H/32*W/32,B,C] 
#         print("1 src:",src.shape)
        
        if len(style_src.shape)==4:
            bs, c, h, w = style_src.shape
#         print("0 style_src:",style_src.shape)
        
#         style_src=F.unfold(style_src,kernel_size=(self.fold_k, self.fold_k), padding=self.fold_p,stride=self.fold_stride) 
        # [B,C*k*k,L]  L=[(H − k + 2P )/S+1] * [(W − k + 2P )/S+1]
    
#         style_src=self.unfold_ours(style_src)
#         style_src=style_src.permute(2,0,1)   #[H*W,B,c*kh*kw]
        style_src = style_src.flatten(2).permute(2, 0, 1)
#         print("1 style_src:",style_src.shape)
#         print("0 pos_embed:",pos_embed.shape)
        
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1) if pos_embed is not None else None#[H*W,B,C]
        query_pos_embed = query_pos_embed.flatten(2).permute(2, 0, 1)  if query_pos_embed is not None else None  #[H*W,B,C]
        mask = mask.flatten(1)
#         print("mask:",mask.shape)
#         tgt = torch.zeros_like(query_embed)
        tgt = src
#         print(style_src.shape, mask.shape, pos_embed.shape)
        memory = self.encoder(style_src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_pos_embed)
#         print("hs:",hs.shape) #torch.Size([layer_num, h*w, B, C])  
        if len(src.shape)==4:
            return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)
        else:
            return hs.transpose(1, 2), memory.permute(1, 2, 0)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


def show_feature_map(feature_map,verbose="",out_root="",norm=False):
    import imageio
    import numpy as np
    import os
    import pandas as pd

    from torchvision.utils import save_image
    print(feature_map.shape)
    if norm:
        feature_map = feature_map*feature_map.shape[0]*feature_map.shape[1]*feature_map.shape[2]
    feature_map_num = feature_map.shape[0]
#     plt.figure()
    for index in range(feature_map_num)[:50]:
#         plt.subplot(row_num, row_num, index)
#         plt.imshow(feature_map[index-1], cmap='gray')
#         plt.axis('off')
        imageio.imwrite(os.path.join(out_root,"{}_{}.png".format(index,verbose)), feature_map[index].cpu().detach().numpy())
        save_image(feature_map[index],os.path.join(out_root,"vision_{}_{}.png".format(index,verbose))  )
        print(  str(pd.DataFrame(feature_map[index].cpu().detach().numpy()).describe())  )  
#         assert(False)
        break
    
class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for li,layer in enumerate(self.layers):
            output,att = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                if self.norm is not None:
                    intermediate.append(self.norm(output))
                else:
                    intermediate.append(output)
            
#             show_feature_map(att,verbose="tgt2_att_{}".format(li),out_root="tmp_out",norm=True)
#             show_feature_map(output.transpose(0,1)[:,:,[0]].reshape,verbose="tgt2_output_{}".format(li),out_root="tmp_out")
            
#             assert(False)
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,enorm=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.enorm = enorm
        if enorm:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
#         print("q.shape, k.shape, src.shape:",q.shape, k.shape, src.shape)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
#         print(self.self_attn(q, k, value=src, attn_mask=src_mask,
#                               key_padding_mask=src_key_padding_mask)[1].shape)
#         assert(False)
        src = src + self.dropout1(src2)
        if self.enorm:
            src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        if self.enorm:
            src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        if self.enorm:
            src2 = self.norm1(src)
        else:
            src2 = src
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        if self.enorm:
            src2 = self.norm2(src)
        else:
            src2 = src
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,dnorm=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.dnorm = dnorm
        if dnorm:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        if self.dnorm:
            tgt = self.norm1(tgt)
#         print("self.with_pos_embed(tgt, query_pos).shape,self.with_pos_embed(memory, pos).shape,memory.shape:",
#               self.with_pos_embed(tgt, query_pos).shape,self.with_pos_embed(memory, pos).shape,memory.shape)
#         tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
#                                    key=self.with_pos_embed(memory, pos),
#                                    value=memory, attn_mask=memory_mask,
#                                    key_padding_mask=memory_key_padding_mask)[0]
        tgt2_ = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt2=tgt2_[0]
        tgt2_att=tgt2_[1]
        
#         print("tgt2.shape",tgt2.shape)
        tgt = tgt + self.dropout2(tgt2)
        if self.dnorm:
            tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        if self.dnorm:
            tgt = self.norm3(tgt)
        return tgt,tgt2_att

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        if self.dnorm:
            tgt2 = self.norm1(tgt)
        else:
            tgt2 = tgt
        
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        if self.dnorm:
            tgt2 = self.norm2(tgt)
        else:
            tgt2 = tgt
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        if self.dnorm:
            tgt2 = self.norm3(tgt)
        else:
            tgt2 = tgt
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        enorm=args.enorm,
        dnorm=args.dnorm,
    )

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
