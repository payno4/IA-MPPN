"""
Copyright 2023 Yao Qiang, Chengyin Li, Hui Zhu,
Prashant Khanduri, Dongxiao Zhu

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

Modifications:
- Modified by Julian Fernando Weber, 2026: modified class for regression task and configs, added comments
"""



# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np
from config.model_configs.IA_ViT_configs import IA_VisionTransformerConfig, IA_VisionTransformerConfig

from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from config.configs import Configs
from config.model_configs.IA_ViT_configs import IA_VisionTransformerConfig

logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": nn.functional.gelu, "relu": nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config: IA_VisionTransformerConfig, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.num_heads
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size) # [B, N, all_head_size]
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.attention_dropout_rate)
        self.proj_dropout = Dropout(config.attention_dropout_rate)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] # [B, N, D] -> [B, N]
        new_x_shape += (self.num_attention_heads, self.attention_head_size) # [B, N, num_heads, head_size]
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) # [B, num_heads, N, head_size]

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states) # [B, N, hidden_states] -> [B, N, all_head_size]
        mixed_key_layer = self.key(hidden_states) # -""-
        mixed_value_layer = self.value(hidden_states) # -""-

        query_layer = self.transpose_for_scores(mixed_query_layer)# [B, N, all_head_size] -> [B, num_heads, N, head_size]
        key_layer = self.transpose_for_scores(mixed_key_layer) # -""-
        value_layer = self.transpose_for_scores(mixed_value_layer) # -""-

        #for each head and for each token, compute attention score to every other token
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) #[B, num_heads, N, head_size] x [B, num_heads, head_size, N] -> [B, num_heads, N, N]
        #stabilize gradients
        attention_scores /= math.sqrt(self.attention_head_size) # [B, num_heads, N, N]
        # differences got enhanced, irrelevant ones suppressed, relevant tokens get higher weights
        attention_probs = self.softmax(attention_scores)# [B, num_heads, N, N]
        attention_probs = self.attn_dropout(attention_probs)
        weights = attention_probs if self.vis else None

        context_layer = torch.matmul(attention_probs, value_layer) # [B, num_heads, N, N] x [B, num_heads, N, head_size] -> [B, num_heads, N, head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # [B, N, num_heads, head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) # [B, N, all_head_size]
        context_layer = context_layer.view(*new_context_layer_shape) # [B, N, all_head_size], concatenate all heads
        attention_output = self.out(context_layer) #[B, N, all_head_size] -> [B, N, hidden_size]
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config:IA_VisionTransformerConfig):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.mlp_dim) #[B, N, hidden_size] -> [B, N, mlp_dim]
        self.fc2 = Linear(config.mlp_dim, config.hidden_size) #[B, N, mlp_dim] -> [B, N, hidden_size]
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.dropout_rate)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        #x: [B, N, D]
        x = self.fc1(x) # [B, N, hidden_size] -> [B, N, mlp_dim]
        x = self.act_fn(x) # [B, N, mlp_dim]
        x = self.dropout(x) 
        x = self.fc2(x) # [B, N, mlp_dim] -> [B, N, hidden_size]
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config: IA_VisionTransformerConfig, img_size, in_channels): # type: ignore
        super(Embeddings, self).__init__()
        img_size = _pair(img_size)
        self.patch_size = None # for easier access later

        self.patch_size = _pair(config.patches["size"])
        self.n_patches = (img_size[0] // self.patch_size[0]) * (img_size[1] // self.patch_size[1])

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=self.patch_size,
                                       stride=self.patch_size) #[B, hidden_size, n_patches_H, n_patches_W]
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches+1, config.hidden_size)) # [1, n_patches+1, hidden_size]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) # [1, 1, hidden_size]

        self.dropout = Dropout(config.dropout_rate)

    def forward(self, x): # type: ignore
        # x -> [B, H, W, C]
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1) # [B, 1, hidden_size]

        x: torch.Tensor = self.patch_embeddings(x) # [B, hidden_size, n_patches_H, n_patches_W]
        x = x.flatten(2) # [B, hidden_size, n_patches_H, n_patches_W] -> [B, hidden_size, n_patches]
        x = x.transpose(-1, -2) #[B, n_patches, hidden_size] #-1 or -2 are dimension counted from the end
        x = torch.cat((cls_tokens, x), dim=1) # [B, n_patches+1, hidden_size]

        embeddings = x + self.position_embeddings # [B, n_patches+1, hidden_size] added position embeddings because it's a learnable parameter for every token
        embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Module):
    def __init__(self, config:IA_VisionTransformerConfig, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x # for residual connection [B, N+1, D]
        x = self.attention_norm(x)
        x, weights = self.attn(x) # [B, N+1, D] -> [B, N+1, D]
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x) # [B, N+1, D] -> [B, N+1, D]
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config: IA_VisionTransformerConfig, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.num_layers):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config: IA_VisionTransformerConfig, vis, input_shape:tuple[int, int, int]):
        super(Transformer, self).__init__()
        (H, _, C) = input_shape
        self.embeddings = Embeddings(config, img_size=H, in_channels=C)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids, return_patch_grid=False):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)

        if not return_patch_grid:
            return encoded, attn_weights
        else:
            B, N_plus, D = encoded.shape
            patch_tokens = encoded[:, 1:, :]

            #grid size
            n_patches = patch_tokens.shape[1]
            H_p = W_p = int(math.sqrt(n_patches))
            assert H_p * W_p == n_patches, "Patch tokens are not a perfect square."

            patch_grid = patch_tokens.view(B, H_p, W_p, D)
            patch_grid = patch_grid.permute(0, 3, 1, 2).contiguous()  # (B, D, H_p, W_p)

            return encoded, attn_weights, patch_grid

class VisionTransformer(nn.Module):
    def __init__(self, config: IA_VisionTransformerConfig, input_shape: tuple[int, int, int], num_patches:int, zero_head=False, vis=False):
        assert isinstance(config, IA_VisionTransformerConfig), "config must be an instance of IA_VisionTransformerConfig"
        self.config = config
        super(VisionTransformer, self).__init__()
        self.num_classes = 1 # changed for regression
        self.zero_head = zero_head
        self.classifier = None # deprecated

        self.transformer = Transformer(config, vis, input_shape=input_shape)
        feature_dim = config.hidden_size
        self.head = nn.Linear(feature_dim, self.num_classes)
        
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)

        self.fc1 = nn.Linear(config.hidden_size, 1)
        self.fc2 = nn.Linear(num_patches, self.num_classes)
        self.softmax = nn.Softmax(dim=-1)

        self._init_weights()


    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)
        
    def forward(self, x, labels=None, return_patch_grid=False, return_patch_tokens=False):
        #x: [B, H, W, C]
        if return_patch_grid:
            x, attn_weights, patch_grid = self.transformer(x, return_patch_grid=True)# [B, N+1, D], attn_weights, [B, D, H_p, W_p]
        else:
            x, attn_weights = self.transformer(x, return_patch_grid=False) # [B, N+1, D], attn_weights

        logits = self.head(x[:, 0]) # [B, D] -> [B, num_classes], CLS Token

        h = x[:, 1:]  # [B, N, D]
        Q = self.query(h)  # [B, N, D]
        K = self.key(h)  # [B, N, D]
        V = self.value(h)  # [B, N, D]

        attn_scores = torch.matmul(Q, K.transpose(-1, -2))  # [B, N, D] x [B, D, N] -> [B, N, N]
        new_atts = self.softmax(attn_scores)  # [B, N, N]
        
        context_layer = torch.matmul(new_atts, V)  # [B, N, N] x [B, N, D] -> [B, N, D]

        inter_logits = self.fc2(self.fc1(context_layer).squeeze(dim=2))

        returns = ()
        returns += (logits, attn_weights, inter_logits, new_atts)
        if return_patch_grid:
            returns += (patch_grid,) # [B, D, H_p, W_p]
        if return_patch_tokens:
            returns += (x[:, 1:],)
        return returns

    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())

            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"])) # type: ignore
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1) #type: ignore
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname) # type: ignore