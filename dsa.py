"""
Differentiable Siamese Augmentation (DSA) from Zhao & Bilen (2021).
Standard augmentation used in small-scale DD evaluation.
Includes: color jitter, crop, cutout, flip, scale, rotate.
"""
import torch
import torch.nn.functional as F
import numpy as np


def DiffAugment(x, strategy='color_crop_cutout_flip_scale_rotate', seed=-1, param=None):
    """Apply DSA augmentations."""
    if seed == -1:
        param.batchmode = False
    else:
        param.batchmode = True
    
    if strategy == 'None' or strategy == '':
        return x
    
    if param is None:
        param = ParamDiffAug()
    
    if seed != -1:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    for p in strategy.split('_'):
        for f in AUGMENT_FNS[p]:
            x = f(x, param)
    
    return x


class ParamDiffAug():
    def __init__(self):
        self.aug_mode = 'S'  # 'S': same augmentation for all images in batch
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5
        self.batchmode = False


def set_seed_DiffAug(param):
    if param.batchmode:
        torch.manual_seed(param.cur_seed)
        np.random.seed(param.cur_seed)


def rand_scale(x, param):
    ratio = param.ratio_scale
    sx = torch.rand(1).item() * (ratio - 1.0/ratio) + 1.0/ratio
    sy = torch.rand(1).item() * (ratio - 1.0/ratio) + 1.0/ratio
    theta = torch.tensor([[[sx, 0, 0], [0, sy, 0]]], dtype=x.dtype, device=x.device)
    theta = theta.repeat(x.shape[0], 1, 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False)
    return x


def rand_rotate(x, param):
    ratio = param.ratio_rotate
    theta_val = (torch.rand(1).item() - 0.5) * 2 * ratio / 180 * np.pi
    cos_t = np.cos(theta_val)
    sin_t = np.sin(theta_val)
    theta = torch.tensor([[[cos_t, -sin_t, 0], [sin_t, cos_t, 0]]], 
                         dtype=x.dtype, device=x.device)
    theta = theta.repeat(x.shape[0], 1, 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False)
    return x


def rand_flip(x, param):
    prob = param.prob_flip
    if torch.rand(1).item() < prob:
        x = x.flip(3)
    return x


def rand_brightness(x, param):
    ratio = param.brightness
    x = x + (torch.rand(1, 1, 1, 1, device=x.device, dtype=x.dtype) - 0.5) * ratio
    return x


def rand_saturation(x, param):
    ratio = param.saturation
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(1, 1, 1, 1, device=x.device, dtype=x.dtype) * ratio) + x_mean
    return x


def rand_contrast(x, param):
    ratio = param.contrast
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(1, 1, 1, 1, device=x.device, dtype=x.dtype) + ratio) + x_mean
    return x


def rand_crop(x, param):
    ratio = param.ratio_crop_pad
    shift_x = int(x.shape[2] * ratio + 0.5)
    shift_y = int(x.shape[3] * ratio + 0.5)
    
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[1]).item()
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[1]).item()
    
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.shape[0], device=x.device),
        torch.arange(x.shape[2], device=x.device),
        torch.arange(x.shape[3], device=x.device),
        indexing='ij'
    )
    grid_x = torch.clamp(grid_x + translation_x, 0, x.shape[2] - 1)
    grid_y = torch.clamp(grid_y + translation_y, 0, x.shape[3] - 1)
    x = x[grid_batch, :, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, param):
    ratio = param.ratio_cutout
    cutout_size = int(x.shape[2] * ratio + 0.5), int(x.shape[3] * ratio + 0.5)
    
    offset_x = torch.randint(0, x.shape[2] + (1 - cutout_size[0] % 2), size=[1]).item()
    offset_y = torch.randint(0, x.shape[3] + (1 - cutout_size[1] % 2), size=[1]).item()
    
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.shape[0], device=x.device),
        torch.arange(cutout_size[0], device=x.device),
        torch.arange(cutout_size[1], device=x.device),
        indexing='ij'
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, 0, x.shape[2] - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, 0, x.shape[3] - 1)
    
    mask = torch.ones(x.shape[0], x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'crop': [rand_crop],
    'cutout': [rand_cutout],
    'flip': [rand_flip],
    'scale': [rand_scale],
    'rotate': [rand_rotate],
}


if __name__ == '__main__':
    x = torch.randn(4, 3, 32, 32).cuda()
    param = ParamDiffAug()
    y = DiffAugment(x, strategy='color_crop_cutout_flip_scale_rotate', param=param)
    print(f"DSA input: {x.shape}, output: {y.shape}")
