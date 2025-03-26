import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register

def expand_batch_dim(param, batch_size):
    """add batch dimension"""
    return param.unsqueeze(0).expand(batch_size, -1, -1)



class ConditionInjectionBlock(nn.Module):
    def __init__(self, inp_feature,color_embedding, texture_embedding, window_size):
        super(ConditionInjectionBlock, self).__init__()
        self.feature = inp_feature
        self.color_embedding = color_embedding # [B, 10, num_gaussians]
        self.texture_embedding = texture_embedding # [B, 7, num_gaussians]
        self.window_size = window_size
    
    



class GaussianMixerBlock(nn.Module):
    def __init__(self, color_embedding, texture_embedding, window_size, scale):
        super(GaussianMixerBlock, self).__init__()


class GaussianDecoder(nn.Module):
    def __init__(self, inp_feature,color_embedding, texture_embedding, window_size, scale):
        super(GaussianDecoder, self).__init__()





@register('gaussian-splatter')
class GaussianSplatter(nn.Module):
    """A module that applies 2D Gaussian splatting to input features."""

    def __init__(self, encoder_spec, dec_spec, window_size = 7, density = 4, hidden_dim=256):
        """
        Initialize the 2D Gaussian Splatter module.
        Args:
            kernel_size (int): The size of the kernel to convert rasterization.
            unfold_row (int): The number of points in the row dimension of the Gaussian grid.
            unfold_column (int): The number of points in the column dimension of the Gaussian grid.
            window_size (int): 窗口大小
            density (float): 控制高斯核的密度
        """
        super(GaussianSplatter, self).__init__()
        self.encoder = models.make(encoder_spec)
        self.feat, self.logits = None, None

        self.coef = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.freq = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.phase = nn.Linear(2, hidden_dim // 2, bias=False)
        self.dec = models.make(dec_spec, args={'in_dim': hidden_dim})

        # Initialize Trainable Parameters
        self.window_size = window_size
        self.density = density
        
        # 计算每个窗口内的高斯核数量
        num_gaussians = int(window_size * window_size * density)
        
        # 初始化可训练的颜色高斯参数
        self.mu_rgb = nn.Parameter(torch.rand(num_gaussians, 3))  # RGB均值 [num_gaussians, 3]
        self.sigma_rgb = nn.Parameter(F.softplus(torch.randn(num_gaussians, 3)*0.5 + 0.1))  # RGB标准差
        self.rho_rgb = nn.Parameter(torch.tanh(torch.rand(num_gaussians, 3) * 0.2))  # RGB相关系数
        self.opacity = nn.Parameter(torch.ones(num_gaussians, 1))  # 不透明度

        # 初始化可训练的纹理高斯参数
        fine_height = int(window_size * density)
        fine_width = int(window_size * density)
        # 生成归一化的网格坐标
        y_coords = torch.linspace(-1, 1, fine_height)
        x_coords = torch.linspace(-1, 1, fine_width)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        kernel_mesh_normalized = torch.stack([xx.flatten(), yy.flatten()], dim=1) # [H*W*density, 2]
        # 初始化纹理高斯参数
        self.mu_tex = nn.Parameter(kernel_mesh_normalized) # [H*W*density, 2]
        self.sigma_x_tex = nn.Parameter(F.softplus(torch.randn_like(self.mu_tex[..., 0]) * 0.5 + 0.2).unsqueeze(-1))  # [num_gaussians, 1]
        self.sigma_y_tex = nn.Parameter(F.softplus(torch.randn_like(self.mu_tex[..., 0]) * 0.5 + 0.2).unsqueeze(-1))  # [num_gaussians, 1]
        self.rho_tex = nn.Parameter(torch.tanh((torch.rand_like(self.mu_tex[..., 0]) * 0.2).unsqueeze(-1)))  # [num_gaussians, 1]
        self.mu_offset = nn.Parameter(torch.ones_like(self.mu_tex))  # [num_gaussians, 2]
        
    def create_texture_gaussian_embedding(self, batch_size):
        """
        创建纹理高斯嵌入
        Args:
            batch_size (int): 批次大小
        Returns:
            texture_gaussian_embedding (torch.Tensor): 形状为 [B, 7, num_gaussians] 的张量
        """
        # num_gaussians = window_size*window_size*density
        return torch.cat([
            expand_batch_dim(self.mu_tex, batch_size), # [B, num_gaussians, 2]
            expand_batch_dim(self.sigma_x_tex, batch_size), # [B, num_gaussians, 1]
            expand_batch_dim(self.sigma_y_tex, batch_size), # [B, num_gaussians, 1]
            expand_batch_dim(self.rho_tex, batch_size), # [B, num_gaussians, 1]
            expand_batch_dim(self.mu_offset, batch_size) # [B, num_gaussians, 1]
        ]).permute(0, 2, 1).contiguous() # [B, 7, num_gaussians]
    
    def create_color_gaussian_embedding(self, batch_size):
        """
        创建颜色高斯嵌入
        Args:
            batch_size (int): 批次大小
        Returns:
            color_gaussian_embedding (torch.Tensor): 形状为 [B, 10, num_gaussians] 的张量
        """
        # 扩展参数到batch维度
        return torch.cat([
            expand_batch_dim(self.mu_rgb, batch_size),      # RGB均值 [B, num_gaussians, 3]
            expand_batch_dim(self.sigma_rgb, batch_size),   # RGB标准差 [B, num_gaussians, 3]
            expand_batch_dim(self.rho_rgb, batch_size),     # RGB相关系数 [B, num_gaussians, 3]
            expand_batch_dim(self.opacity, batch_size)      # 不透明度 [B, num_gaussians, 1]
        ], dim=-1).permute(0, 2, 1).contiguous()  # [B, 10, num_gaussians]



