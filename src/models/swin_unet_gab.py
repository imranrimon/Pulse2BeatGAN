import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
from .swin_unet import (
    SwinTransformerSys, PatchEmbed, BasicLayer, PatchMerging, BasicLayer_up, PatchExpand, 
    FinalPatchExpand_X4, FinalPatchExpand_X2, Mlp
)

class RGuidedAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1x1 = nn.Conv1d(in_channels, out_channels=1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.batch_norm = nn.BatchNorm1d(in_channels)
        
    def forward(self, x, attn_mask, l1_loss=False):
        attn = self.conv1x1(x)
        attn = self.sigmoid(attn)
        if l1_loss:
            l1_loss = F.l1_loss(attn, attn_mask)
        else:
            l1_loss = None
        respath_mult = x * attn
        respath_mult = self.batch_norm(respath_mult)
        out = respath_mult + x
        return out, l1_loss
    
class OGuidedAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1x1 = nn.Conv1d(in_channels, out_channels=1, kernel_size=1)
        self.tanh = nn.Tanh()
        self.batch_norm = nn.BatchNorm1d(in_channels)
        
    def forward(self, x, attn_mask, l1_loss=False):
        attn = self.conv1x1(x)
        attn = self.tanh(attn)
        if l1_loss:
            l1_loss = F.l1_loss(attn, attn_mask)
        else:
            l1_loss = None
        respath_mult = x * attn
        respath_mult = self.batch_norm(respath_mult)
        out = respath_mult + x
        return out, l1_loss

class SwinTransformerSysGAB(nn.Module):
    def __init__(self, img_size = 128*4, patch_size=4, in_chans=1, num_classes=1,
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", pretrained_window_sizes=[0, 0, 0, 0], **kwargs):
        super().__init__()

        self.img_size = img_size
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        self.patch_embed = PatchEmbed(
            img_size=img_size[-1], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Guided Attention Blocks
        self.r_guided_attention = RGuidedAttentionBlock(in_channels=embed_dim)
        self.o_guided_attention = OGuidedAttentionBlock(in_channels=embed_dim)
        self.norm_attention = nn.LayerNorm(embed_dim)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution// (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer))+int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),int(embed_dim * 2 ** (self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(patches_resolution // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                         input_resolution=(patches_resolution // (2 ** (self.num_layers - 1 - i_layer))),
                                         depth=depths[(self.num_layers - 1 - i_layer)],
                                         num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                         window_size=window_size,
                                         mlp_ratio=self.mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop_rate, attn_drop=attn_drop_rate,
                                         drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                             depths[:(self.num_layers - 1 - i_layer) + 1])],
                                         norm_layer=norm_layer,
                                         upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                         use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
       
        self.norm_up_512 = norm_layer(self.embed_dim)
        self.norm_up_256= norm_layer(self.embed_dim*2)
        self.norm_up_128 = norm_layer(self.embed_dim*4)
        
        if self.final_upsample == "expand_first":
            self.up_128 = FinalPatchExpand_X2(input_resolution=img_size[0] // patch_size,
                                          dim_scale=2, dim=embed_dim*4)
            self.up_256 = FinalPatchExpand_X2(input_resolution=img_size[1] // patch_size,
                                          dim_scale=2, dim=embed_dim*2)
                                        
            self.up_512 = FinalPatchExpand_X4(input_resolution=img_size[2] // patch_size,
                                          dim_scale=4, dim=embed_dim)
            self.output_128 = nn.Conv1d(in_channels=embed_dim*4*8, out_channels=self.num_classes, kernel_size=1, bias=False)
            self.output_256 = nn.Conv1d(in_channels=embed_dim*4*4, out_channels=self.num_classes, kernel_size=1, bias=False)
            self.output_512 = nn.Conv1d(in_channels=embed_dim*4, out_channels=self.num_classes, kernel_size=1, bias=False)

        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)  
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []

        for inx, layer in enumerate(self.layers):
            x_downsample.append(x)
            x = layer(x)
        x = self.norm(x)

        return x, x_downsample

    def forward_up_features(self, x, x_downsample, norm_up, rpeaks_mask, opeaks_mask, use_l1_loss):
        out = []
        l1_losses = []
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[3 - inx]], dim=-1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)
                
                if inx == len(self.layers_up) - 1:
                    # Interpolate masks to match feature map size
                    current_len = x.shape[1]
                    if rpeaks_mask.shape[-1] != current_len:
                        rpeaks_mask_resized = F.interpolate(rpeaks_mask, size=current_len, mode='nearest')
                    else:
                        rpeaks_mask_resized = rpeaks_mask
                        
                    if opeaks_mask.shape[-1] != current_len:
                        opeaks_mask_resized = F.interpolate(opeaks_mask, size=current_len, mode='nearest')
                    else:
                        opeaks_mask_resized = opeaks_mask

                    x_r, l1_loss_r = self.r_guided_attention(x.permute(0, 2, 1), rpeaks_mask_resized, use_l1_loss)  
                    x_o, l1_loss_o = self.o_guided_attention(x.permute(0, 2, 1), opeaks_mask_resized, use_l1_loss)  
                    x = (x_r + x_o).permute(0, 2, 1)

                    l1_losses.append(l1_loss_r)
                    l1_losses.append(l1_loss_o)

                out.append(norm_up[inx - 1](x))

        return out, l1_losses
    
    def up_x2(self, x, up, output, H= 128):
        B, L, C = x.shape
        if self.final_upsample == "expand_first":
            x = up(x)
            x = x.view(B, 2 * H, -1)
            x = x.permute(0, 2, 1)
            x = output(x)
        return x
    
    def up_x4(self, x, up, output):
        H = self.patches_resolution
        B, L, C = x.shape
        if self.final_upsample == "expand_first":
            x = up(x)
            x = x.view(B, 4 * H, -1)
            x = x.permute(0, 2, 1)
            x = output(x)
        return x

    def forward(self, x, rpeaks_mask, opeaks_mask, use_l1_loss=False):
        x, x_downsample = self.forward_features(x)
        x, l1_losses = self.forward_up_features(x, x_downsample, norm_up=[self.norm_up_128, self.norm_up_256, self.norm_up_512], rpeaks_mask=rpeaks_mask, opeaks_mask=opeaks_mask, use_l1_loss=use_l1_loss)

        x_128 = self.up_x2(x[0], self.up_128, self.output_128, H=64)
        x_256 = self.up_x2(x[1], self.up_256, self.output_256, H=128)
        x_512 = self.up_x4(x[2], self.up_512, self.output_512)
        return x_128, x_256, x_512, l1_losses

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution// (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
