# # Reference: https://github.com/tljxyys/GaussianSR

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

import models
from models import register
from utils import to_pixel_samples



def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers."""
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def generate_meshgrid(height, width):
    """
    Generate a meshgrid of coordinates for a given image dimensions.
    Args:
        height (int): Height of the image.
        width (int): Width of the image.
    Returns:
        torch.Tensor: A tensor of shape [height * width, 2] containing the (x, y) coordinates for each pixel in the image.
    """
    # Generate all pixel coordinates for the given image dimensions
    y_coords, x_coords = torch.arange(0, height), torch.arange(0, width)
    # Create a grid of coordinates
    yy, xx = torch.meshgrid(y_coords, x_coords)
    # Flatten and stack the coordinates to obtain a list of (x, y) pairs
    all_coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    return all_coords


def fetching_features_from_tensor(image_tensor, input_coords):
    """
    Extracts pixel values from a tensor of images at specified coordinate locations.
    Args:
        image_tensor (torch.Tensor): A 4D tensor of shape [batch, channel, height, width] representing a batch of images.
        input_coords (torch.Tensor): A 2D tensor of shape [N, 2] containing the (x, y) coordinates at which to extract pixel values.
    Returns:
        color_values (torch.Tensor): A 3D tensor of shape [batch, N, channel] containing the pixel values at the specified coordinates.
        coords (torch.Tensor): A 2D tensor of shape [N, 2] containing the normalized coordinates in the range [-1, 1].
    """
    # Normalize pixel coordinates to [-1, 1] range
    input_coords = input_coords.to(image_tensor.device)
    coords = input_coords / torch.tensor([image_tensor.shape[-2], image_tensor.shape[-1]],
                                         device=image_tensor.device).float()
    center_coords_normalized = torch.tensor([0.5, 0.5], device=image_tensor.device).float()
    coords = (center_coords_normalized - coords) * 2.0

    # Fetching the color of the pixels in each coordinates
    batch_size = image_tensor.shape[0]
    input_coords_expanded = input_coords.unsqueeze(0).expand(batch_size, -1, -1) # B, N, 2

    # 提取x, y坐标
    y_coords = input_coords_expanded[..., 0].long()
    x_coords = input_coords_expanded[..., 1].long()
    batch_indices = torch.arange(batch_size).view(-1, 1).to(input_coords.device) # B, 1

    color_values = image_tensor[batch_indices, :, x_coords, y_coords]

    return color_values, coords


def extract_patch(image, center, radius, padding_mode='constant'):
    """
    Extract a patch from an image with the specified center and radius.
    Args:
        image (torch.Tensor): Input image of shape [batch_size, channels, height, width].
        center (tuple): Coordinates (y, x) of the patch center.
        radius (int): Radius of the patch.
        padding_mode (str, optional): Padding mode, can be 'constant', 'reflect', 'replicate', or 'circular'. Default is 'constant'.

    Returns:
        torch.Tensor: Extracted patch of shape [batch_size, channels, 2 * radius, 2 * radius].
    """
    height, width = image.shape[-2:]

    # Convert center coordinates to integers
    center_y, center_x = int(round(center[0])), int(round(center[1]))

    # Calculate patch boundaries
    top = center_y - radius
    bottom = center_y + radius
    left = center_x - radius
    right = center_x + radius

    # Check if boundaries are out of image bounds
    top_padding = max(0, -top)
    bottom_padding = max(0, bottom - height)
    left_padding = max(0, -left)
    right_padding = max(0, right - width)

    # Pad the image
    padded_image = torch.nn.functional.pad(image, (left_padding, right_padding, top_padding, bottom_padding),
                                           mode=padding_mode)

    # Extract the patch
    patch = padded_image[..., top_padding:top_padding + 2 * radius, left_padding:left_padding + 2 * radius]

    return patch


@register('gaussian-splatter')
class GaussianSplatter(nn.Module):
    """A module that applies 2D Gaussian splatting to input features."""

    def __init__(self, encoder_spec, dec_spec, kernel_size, hidden_dim=256, unfold_row=7, unfold_column=7,
                 num_points=100):
        """
        Initialize the 2D Gaussian Splatter module.
        Args:
            kernel_size (int): The size of the kernel to convert rasterization.
            unfold_row (int): The number of points in the row dimension of the Gaussian grid.
            unfold_column (int): The number of points in the column dimension of the Gaussian grid.
        """
        super(GaussianSplatter, self).__init__()
        self.encoder = models.make(encoder_spec)
        self.feat, self.logits = None, None

        self.coef = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.freq = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.phase = nn.Linear(2, hidden_dim // 2, bias=False)
        self.dec = models.make(dec_spec, args={'in_dim': hidden_dim})

        # Key parameter in 2D Gaussian Splatter
        self.kernel_size = kernel_size
        self.row = unfold_row
        self.column = unfold_column
        self.num_points = num_points

        # Initialize Trainable Parameters
        sigma_x, sigma_y = torch.meshgrid(torch.linspace(0.2, 3.0, 10), torch.linspace(0.2, 3.0, 10))
        self.sigma_x = sigma_x.reshape(-1)
        self.sigma_y = sigma_y.reshape(-1)
        self.opacity = torch.sigmoid(torch.ones(self.num_points, 1, requires_grad=True)) # [100, 1]
        self.rho = torch.clamp(torch.zeros(self.num_points, 1, requires_grad=True), min=-1, max=1) # 相关系数
        self.sigma_x = nn.Parameter(self.sigma_x)  # Standard deviation in x-axis
        self.sigma_y = nn.Parameter(self.sigma_y)  # Standard deviation in y-axis
        self.opacity = nn.Parameter(self.opacity)  # Transparency of feature, shape=[num_points, 1]
        self.rho = nn.Parameter(self.rho)

    def create_texture_gaussian_embedding(self, image_tensor_shape, density=4.0):
        """
        Create a texture Gaussian embedding for the input image tensor.
        Args:
            image_tensor_shape (tuple): Shape of input tensor [batch, channel, height, width].
            density (float): Control the density of the Gaussian grid points. Default: 4.0.
        Returns:
            gaussian_embedding (torch.Tensor): Tensor of shape [batch, 7, H*W*density] containing:
                - mu: normalized coordinates [-1, 1] for each point
                - sigma_x: x-axis standard deviation
                - sigma_y: y-axis standard deviation 
                - rho: correlation coefficient
                - mu_offset: offset for coordinates
        """
        # Generate a meshgrid of coordinates 
        batch_size, _, height, width = image_tensor_shape
        fine_height = int(height * density)
        fine_width = int(width * density)

        # 生成归一化的网格坐标
        y_coords = torch.linspace(-1, 1, fine_height)
        x_coords = torch.linspace(-1, 1, fine_width)
        yy, xx = torch.meshgrid(y_coords, x_coords)
        kernel_mesh_normalized = torch.stack([xx.flatten(), yy.flatten()], dim=1) # [H*W*density, 2]
        
        # 初始化高斯参数
        mu = kernel_mesh_normalized.unsqueeze(0).expand(batch_size, -1, -1)  # [B, H*W*density, 2]
        sigma_x = torch.randn_like(mu[..., 0]).clamp(0.2, 3.0).unsqueeze(-1)  # [B, H*W*density, 1]
        sigma_y = torch.randn_like(mu[..., 0]).clamp(0.2, 3.0).unsqueeze(-1)  # [B, H*W*density, 1]
        rho = (torch.rand_like(mu[..., 0]) * 2 - 1).clamp(-1, 1).unsqueeze(-1)  # [B, H*W*density, 1]
        mu_offset = torch.ones_like(mu)  # [B, H*W*density, 2]

        # 组装高斯参数
        self.texture_gaussian_embedding = torch.cat([
            mu,
            sigma_x,
            sigma_y,
            rho,
            mu_offset
        ], dim=-1).permute(0, 2, 1)  # [B, 7, H*W*density]

        return self.texture_gaussian_embedding


    def create_color_gaussian_embedding(self, image_tensor, density=4.0):
        """
        Create a color Gaussian embedding for the input image tensor.
        Args:
            image_tensor (torch.Tensor): Input image tensor of shape [batch, channel, height, width].
            density (float): Control the density of the Gaussian kernel. Default: 4.0.
        Returns:
            color_gaussian_embedding: a tensor with shape :[B, 10, H*W*density], which containing:
                - mu_rgb: RGB mean values [B, 3, H*W*density]
                - sigma_rgb: RGB standard deviations [B, 3, H*W*density]
                - rho_rgb: RGB correlation coefficients [B, 3, H*W*density]
                - opacity: Transparency values [B, 1, H*W*density]
        """
        # Generate meshgrid coordinates
        batch_size, _, height, width = image_tensor.shape
        fine_height = int(height * density)
        fine_width = int(width * density)
        
        # 生成归一化的网格坐标
        y_coords = torch.linspace(-1, 1, fine_height)
        x_coords = torch.linspace(-1, 1, fine_width)
        yy, xx = torch.meshgrid(y_coords, x_coords)
        kernel_mesh_normalized = torch.stack([xx.flatten(), yy.flatten()], dim=1) # [H*W*density, 2]
        
        # 对原始图像进行采样获取RGB值作为均值
        coords = kernel_mesh_normalized.unsqueeze(0).expand(batch_size, -1, -1)  # [B, H*W*density, 2]
        coords_sample = coords.flip(-1).unsqueeze(1)  # [B, 1, H*W*density, 2]
        mu_rgb = F.grid_sample(
            image_tensor, 
            coords_sample,
            mode='bilinear',
            align_corners=False
        ).squeeze(2).permute(0, 2, 1)  # [B, H*W*density, 3]
        
        # 初始化RGB通道的标准差 (使用正态分布,并确保为正)
        sigma_rgb = torch.abs(torch.randn_like(mu_rgb)).clamp(0.1, 1.0)  # [B, H*W*density, 3]
        
        # 初始化RGB通道间的相关系数矩阵 (对称矩阵,对角线为1)
        # 我们需要3个相关系数: rho_rg, rho_rb, rho_gb
        rho_rgb = torch.zeros(batch_size, kernel_mesh_normalized.shape[0], 3, device=coords.device)  # [B, H*W*density, 3]
        rho_rgb[..., 0] = (torch.rand_like(rho_rgb[..., 0]) * 2 - 1).clamp(-1, 1)  # rho_rg
        rho_rgb[..., 1] = (torch.rand_like(rho_rgb[..., 1]) * 2 - 1).clamp(-1, 1)  # rho_rb
        rho_rgb[..., 2] = (torch.rand_like(rho_rgb[..., 2]) * 2 - 1).clamp(-1, 1)  # rho_gb
        
        # 初始化不透明度
        opacity = torch.ones(batch_size, kernel_mesh_normalized.shape[0], 1, device=coords.device)  # [B, H*W*density, 1]
        
        # 组装所有参数
        self.color_gaussian_embedding = torch.cat([
            mu_rgb,          # RGB均值 [B, H*W*density, 3]
            sigma_rgb,       # RGB标准差 [B, H*W*density, 3]
            rho_rgb,         # RGB相关系数 [B, H*W*density, 3]
            opacity,         # 不透明度 [B, H*W*density, 1]
        ], dim=-1).permute(0, 2, 1)  # [B, 10, H*W*density]
        
        return self.color_gaussian_embedding


    def weighted_gaussian_parameters(self, logits):
        """
        Computes weighted Gaussian parameters based on logits and the Gaussian kernel parameters (sigma_x, sigma_y, opacity).
        The logits tensor is used as a weight to compute a weighted sum of the Gaussian kernel parameters for each spatial
        location across the batch dimension. The resulting weighted parameters are then averaged across the batch dimension.
        Args:
            logits (torch.Tensor): Logits tensor of shape [batch, class, height, width].
        Returns:
            tuple: A tuple containing the weighted Gaussian parameters:
                - weighted_sigma_x (torch.Tensor): Tensor of shape [height * width] representing the weighted x-axis standard deviations.
                - weighted_sigma_y (torch.Tensor): Tensor of shape [height * width] representing the weighted y-axis standard deviations.
                - weighted_opacity (torch.Tensor): Tensor of shape [height * width] representing the weighted opacities.
        Description:
            This function computes weighted Gaussian parameters based on the input tensor, logits, and the provided Gaussian kernel parameters (sigma_x, sigma_y, and opacity). The logits tensor is used as a weight to compute a weighted sum of the Gaussian kernel parameters for each spatial location (height and width) across the batch dimension. The resulting weighted parameters are then averaged across the batch dimension, yielding tensors of shape [height * width] for the weighted sigma_x, sigma_y, and opacity.
        """
        batch_size, num_classes, height, width = logits.size()
        logits = logits.permute(0, 2, 3, 1)  # Reshape logits to [batch, height, width, class]

        # Compute weighted sum of Gaussian parameters across class dimension
        weighted_sigma_x = (logits * self.sigma_x.unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum(dim=-1) # [B, H, W]
        weighted_sigma_y = (logits * self.sigma_y.unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum(dim=-1) # [B, H, W]
        weighted_opacity = (logits * self.opacity[:, 0].unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum(dim=-1) # [B, H, W]
        weighted_rho = (logits * self.rho[:, 0].unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum(dim=-1) # [B, H, W]

        # Reshape and average across batch dimension
        weighted_sigma_x = weighted_sigma_x.reshape(batch_size, -1).mean(dim=0) # [B, H, W] => [B, H*w] => [H*W] 
        weighted_sigma_y = weighted_sigma_y.reshape(batch_size, -1).mean(dim=0) # 消去了维度batch_size
        weighted_opacity = weighted_opacity.reshape(batch_size, -1).mean(dim=0)
        weighted_rho = weighted_rho.reshape(batch_size, -1).mean(dim=0)

        return weighted_sigma_x, weighted_sigma_y, weighted_opacity, weighted_rho # 形状全部为 [H*W]

    def gen_feat(self, inp):
        """Generate feature and logits by encoder."""
        self.inp = inp
        self.feat, self.logits = self.encoder(inp)
        self.feat_coord = make_coord(inp.shape[-2:], flatten=False).cuda().permute(2, 0, 1) \
            .unsqueeze(0).expand(inp.shape[0], 2, *inp.shape[-2:]) # [B, 2, h, w]
        return self.feat, self.logits

    def query_rgb(self, coord, scale, cell=None):
        """
        Continuous sampling through 2D Gaussian Splatting.
        Args:
            coord (torch.Tensor): [Batch, Sample_q, 2]. The normalized coordinates of HR space (of range [-1, 1]).
            cell (torch.Tensor): [Batch, Sample_q, 2]. The normalized cell size of HR space.
            scale (torch.Tensor): [Batch]. The magnification scale of super-resolution. (1, 4) during training.
        Returns:
            torch.Tensor: The output features after Gaussian splatting, of the same shape as the input.
        """
        # 1. Get LR feature and logits
        feat, lr_feat, logits = self.feat[:, :8, :, :], self.feat[:, 8:, :, :], self.logits  # Channel decoupling
        feat_size, feat_device = feat.shape, feat.device

        # 2. Calculate the high-resolution image size
        scale = float(scale[0])
        hr_h = round(feat.shape[-2] * scale)  # shape: [batch size]
        hr_w = round(feat.shape[-1] * scale)

        # 3. Unfold the feature / logits to many small patches to avoid extreme GPU memory consumption
        num_patches_row = math.ceil(feat_size[-2] / self.row)
        num_patches_column = math.ceil(feat_size[-1] / self.column)
        upsampled_size = (num_patches_row * self.row, num_patches_column * self.column)
        upsampled_inp = F.interpolate(feat, size=upsampled_size, mode='bicubic', align_corners=False)
        upsampled_logits = F.interpolate(logits, size=upsampled_size, mode='bicubic', align_corners=False)
        unfold = nn.Unfold(kernel_size=(self.row, self.column), stride=(self.row, self.column))
        unfolded_feature = unfold(upsampled_inp)
        unfolded_logits = unfold(upsampled_logits)
        # Unfolded_feature dimension becomes [Batch, C * row * column, L], where L is the number of columns after unfolding
        L = unfolded_feature.shape[-1]
        unfolded_feature_reshaped = unfolded_feature.transpose(1, 2). \
            reshape(feat_size[0] * L, feat_size[1], self.row, self.column)
        unfold_feat = unfolded_feature_reshaped  # shape: [num of patch * batch, channel, self.row, self.column]
        unfolded_logits_reshaped = unfolded_logits.transpose(1, 2). \
            reshape(logits.shape[0] * L, logits.shape[1], self.row, self.column)
        unfold_logits = unfolded_logits_reshaped  # shape: [num of patch * batch, channel, self.row, self.column]

        # 4. Generate colors_(features) and coords_norm(归一化的坐标)
        coords_ = generate_meshgrid(unfold_feat.shape[-2], unfold_feat.shape[-1])
        num_LR_points = unfold_feat.shape[-2] * unfold_feat.shape[-1] #  self.row * self.column 每个patch内有多少点
        colors_, coords_norm = fetching_features_from_tensor(unfold_feat, coords_) # 获取特征图对应坐标处的value及归一化的坐标

        # 5. Rasterization: Generating grid
        # 5.1. Spread Gaussian points over the whole feature map
        batch_size, channel, _, _ = unfold_feat.shape
        weighted_sigma_x, weighted_sigma_y, weighted_opacity, weighted_rho = self.weighted_gaussian_parameters(
            unfold_logits) #[row * column]
        sigma_x = weighted_sigma_x.view(num_LR_points, 1, 1) # [num_LR_points, 1, 1]
        sigma_y = weighted_sigma_y.view(num_LR_points, 1, 1)
        rho = weighted_rho.view(num_LR_points, 1, 1)

        # 5.2. Gaussian expression
        covariance = torch.stack(
            [torch.stack([sigma_x ** 2 + 1e-5, rho * sigma_x * sigma_y], dim=-1),
             torch.stack([rho * sigma_x * sigma_y, sigma_y ** 2 + 1e-5], dim=-1)], dim=-2
        )  # when correlation rou is set to zero, covariance will always be positive semi-definite
        covariance = covariance.contiguous()
        if torch.det(covariance).abs().min() < 1e-6:
            covariance += torch.eye(covariance.shape[-1], device=covariance.device) * 1e-6
        inv_covariance = torch.inverse(covariance).to(feat_device) # [num_LR_points, 2, 2]

        # 5.3. Choosing a broad range for the distribution [-5,5] to avoid any clipping
        start = torch.tensor([-5.0], device=feat_device).view(-1, 1) # [1, 1]
        end = torch.tensor([5.0], device=feat_device).view(-1, 1)
        base_linspace = torch.linspace(0, 1, steps=self.kernel_size, device=feat_device) # [kernel_size]
        ax_batch = start + (end - start) * base_linspace # 值范围从[0, 1]变为[-5, 5] 形状为 [1, kernel_size]
        # Expanding dims for broadcasting
        ax_batch_expanded_x = ax_batch.unsqueeze(-1).expand(-1, -1, self.kernel_size) # [1, kernel_size, kernel_size]
        ax_batch_expanded_y = ax_batch.unsqueeze(1).expand(-1, self.kernel_size, -1) # [1, kernel_size, kernel_size]

        # 5.4. Creating a batch-wise meshgrid using broadcasting
        xx, yy = ax_batch_expanded_x, ax_batch_expanded_y
        xy = torch.stack([xx, yy], dim=-1) # [1, kernel_size, kernel_size, 2]
        z = torch.einsum('b...i,b...ij,b...j->b...', xy, -0.5 * inv_covariance, xy)
        kernel = torch.exp(z) / (2 * torch.tensor(np.pi, device=feat_device) *
                                 torch.sqrt(torch.det(covariance)).to(feat_device).view(num_LR_points, 1, 1)) # [num_LR_points, kernel_size, kernel_size]
        # 找到高斯核中的最大值，用于归一化
        kernel_max_1, _ = kernel.max(dim=-1, keepdim=True)  # Find max along the last dimension
        kernel_max_2, _ = kernel_max_1.max(dim=-2, keepdim=True)  # Find max along the second-to-last dimension
        kernel_normalized = kernel / kernel_max_2
        # [num_LR_points, channel * kernel_size, kernel_size] => [num_LR_points * channel, kernel_size, kernel_size]
        kernel_reshaped = kernel_normalized.repeat(1, channel, 1).contiguous(). \
            view(num_LR_points * channel, self.kernel_size, self.kernel_size)
        kernel_color = kernel_reshaped.unsqueeze(0).reshape(num_LR_points, channel, self.kernel_size, self.kernel_size) #  [num_LR_points, channel, kernel_size, kernel_size]

        # 5.5. Adding padding to make kernel size equal to the image size => hr image patches
        pad_h = round(unfold_feat.shape[-2] * scale) - self.kernel_size
        pad_w = round(unfold_feat.shape[-1] * scale) - self.kernel_size
        if pad_h < 0 or pad_w < 0:
            raise ValueError("Kernel size should be smaller or equal to the image size.")
        padding = (pad_w // 2, pad_w // 2 + pad_w % 2, pad_h // 2, pad_h // 2 + pad_h % 2)
        kernel_color_padded = torch.nn.functional.pad(kernel_color, padding, "constant", 0)

        # 5.6. Create a batch of 2D affine matrices
        b, c, h, w = kernel_color_padded.shape  # num_LR_points, channel, hr_h, hr_w
        theta = torch.zeros(batch_size, b, 2, 3, dtype=torch.float32, device=feat_device) # [batch_size, num_LR_points, 2, 3]
        theta[:, :, 0, 0] = 1.0
        theta[:, :, 1, 1] = 1.0
        theta[:, :, :, 2] = coords_norm
        grid = F.affine_grid(theta.view(-1, 2, 3), size=[batch_size * b, c, h, w], align_corners=True).contiguous() # [batch_size * num_LR_points, h, w, 2]
        kernel_color_padded_expanded = kernel_color_padded.repeat(batch_size, 1, 1, 1).contiguous() #  [batch_size * num_LR_points, channel, h, w]
        kernel_color_padded_translated = F.grid_sample(kernel_color_padded_expanded.contiguous(), grid.contiguous(),
                                                       align_corners=True)
        kernel_color_padded_translated = kernel_color_padded_translated.view(batch_size, b, c, h, w) # [batch_size, num_LR_points, channel, h, w] h = hr_h

        # 6. Apply Gaussian splatting
        # colors_.shape = [batch, num_LR_points, channel], colors.shape = [batch, num_LR_points, channel]
        colors = colors_ * weighted_opacity.to(feat_device).unsqueeze(-1).expand(batch_size, -1, -1)
        color_values_reshaped = colors.unsqueeze(-1).unsqueeze(-1) # [batch, num_LR_points, channel, 1, 1]
        final_image_layers = color_values_reshaped * kernel_color_padded_translated
        final_image = final_image_layers.sum(dim=1) # [batch_size, channel, h, w]
        final_image = torch.clamp(final_image, 0, 1)

        # 7. Fold the input back to the original size
        # Calculate the number of kernels needed to cover each dimension.
        kernel_h, kernel_w = round(self.row * scale), round(self.column * scale)
        fold = nn.Fold(output_size=(kernel_h * num_patches_row, kernel_w * num_patches_column),
                       kernel_size=(kernel_h, kernel_w), stride=(kernel_h, kernel_w))
        final_image = final_image.reshape(feat_size[0], L, feat_size[1] * kernel_h * kernel_w).transpose(1, 2) # [batch_size, num_of_kernel, C * kernel_size * kernel_size]
        final_image = fold(final_image)
        final_image = F.interpolate(final_image, size=(hr_h, hr_w), mode='bicubic', align_corners=False)
        
        # Combine channel 合并两个分支，一个经过高斯处理，另一个经过双三次上采样得到
        lr_feat = F.interpolate(lr_feat, size=(hr_h, hr_w), mode='bicubic', align_corners=False)
        final_image = torch.concat((final_image, lr_feat), dim=1)

        # 8. Augmentation (Useful for improving out-of-distribution scale performance)
        coef = self.coef(final_image)
        freq = self.freq(final_image)
        feat_coord = self.feat_coord
        coord_ = coord.clone()
        q_coef = F.grid_sample(coef, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        q_freq = F.grid_sample(freq, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                  :] \
            .permute(0, 2, 1)
        # 计算高分像素与最临近的低分坐标的相对坐标
        rel_coord = coord - q_coord # [Batch, Sample_q, 2]
        rel_coord[:, :, 0] *= feat.shape[-2]
        rel_coord[:, :, 1] *= feat.shape[-1]
        rel_cell = cell.clone()
        rel_cell[:, :, 0] *= feat.shape[-2]
        rel_cell[:, :, 1] *= feat.shape[-1]
        bs, q = coord.shape[:2]
        q_freq = torch.stack(torch.split(q_freq, 2, dim=-1), dim=-1)
        q_freq = torch.mul(q_freq, rel_coord.unsqueeze(-1))
        q_freq = torch.sum(q_freq, dim=-2) # [bs, q, freq_dim//2]
        q_freq += self.phase(rel_cell.view((bs * q, -1))).view(bs, q, -1)
        q_freq = torch.cat((torch.cos(np.pi * q_freq), torch.sin(np.pi * q_freq)), dim=-1)

        inp = torch.mul(q_coef, q_freq)

        pred = self.dec(inp.contiguous().view(bs * q, -1)).view(bs, q, -1)

        return pred

    def forward(self, inp, coord, scale, cell=None):
        self.gen_feat(inp)
        return self.query_rgb(coord, scale, cell)


if __name__ == '__main__':
    # A simple example of implementing class GaussianSplatter
    model = GaussianSplatter(encoder_spec={"name": "edsr-baseline", "args": {"no_upsampling": True}},
                             dec_spec={"name": "mlp", "args": {"out_dim": 3, "hidden_list": [256, 256, 256, 256]}},
                             kernel_size=3)
    input = torch.rand(1, 3, 64, 64)
    sr_scale = 2
    hr_coord, hr_rgb = to_pixel_samples(
        F.interpolate(input, size=(round(input.shape[-2] * sr_scale), round(input.shape[-1] * sr_scale)),
                      mode='bicubic', align_corners=False))
    v0_x, v1_x, v0_y, v1_y = -1, 1, -1, 1
    nx, ny = round(input.shape[-2] * sr_scale), round(input.shape[-1] * sr_scale)

    x = ((hr_coord[..., 0] - v0_x) / (v1_x - v0_x) * 2 * (nx - 1) / 2).round().long()
    y = ((hr_coord[..., 1] - v0_y) / (v1_y - v0_y) * 2 * (ny - 1) / 2).round().long()
    restored_coords = torch.stack([x, y], dim=-1)

    sample_lst = np.random.choice(len(hr_coord), 2304, replace=False)
    hr_coord = hr_coord[sample_lst]
    hr_rgb = hr_rgb[sample_lst]
    cell_ = torch.ones_like(hr_coord.unsqueeze(0))
    cell_[:, 0] *= 2 / nx
    cell_[:, 1] *= 2 / ny
    sr_scale = 2 * torch.ones(1)
    print(model(input, hr_coord.unsqueeze(0), sr_scale, cell_).shape)
