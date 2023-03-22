import numpy as np
import albumentations
import torchvision.transforms.functional as FT
import PIL
import torch 
from itertools import product

def deform(img, points, sigma):
    # convert to numpy
    img = np.array(img)
    # apply deformation
    img = albumentations.geometric.transforms.ElasticTransform(sigma=sigma, always_apply=True, approximate=True)(image=img)['image']
    # return PIL image
    return PIL.Image.fromarray(img)

FT.deform = deform


class ManualTransform(object):
    def __init__(self, name, k, norm, resize=256, crop_size=224):
        self.norm = norm
        name_to_fn = {
            'rotation': 'rotate',
            'translation': 'affine',
            'scale': 'affine',
            'shear': 'affine',
            'resized_crop': 'resized_crop',
            'h_flip': 'hflip',
            'v_flip': 'vflip',
            'deform': 'deform',
            'grayscale': 'rgb_to_grayscale',
            'brightness': 'adjust_brightness',
            'contrast': 'adjust_contrast',
            'saturation': 'adjust_saturation',
            'hue': 'adjust_hue',
            'blur': 'gaussian_blur',
            'sharpness': 'adjust_sharpness',
            'invert': 'invert',
            'equalize': 'equalize',
            'posterize': 'posterize',
        }
        self.fn = name_to_fn[name]
        self.k = k
        self.resize = resize
        self.crop_size = crop_size
        if name == 'rotation':
            self.param_keys = ['angle']
            self.param_vals = [torch.linspace(0, 360, self.k + 1)[:self.k].to(torch.float32).tolist()]
            self.original_idx = 0
        elif name == 'translation':
            self.param_keys = ['translate', 'angle', 'scale', 'shear']
            space = (1 - (crop_size / resize)) / 2
            a = torch.linspace(-space, space, int(np.sqrt(self.k)) + 1)[:int(np.sqrt(self.k))].to(torch.float32)
            translate_params = [(float(a * resize), float(b * resize)) for a, b in product(a, a)]
            self.param_vals = [
                translate_params,
                torch.zeros(self.k).tolist(),
                torch.ones(self.k).tolist(),
                torch.ones(self.k).tolist()
            ]
            self.original_idx = translate_params.index((0.0, 0.0))
        elif name == 'scale':
            self.param_keys = ['scale', 'translate', 'angle', 'shear']
            self.param_vals = [
                torch.linspace(1 / 4, 2, self.k).to(torch.float32).tolist(),
                torch.zeros((self.k, 2)).tolist(),
                torch.zeros(self.k).tolist(),
                torch.ones(self.k).tolist()
            ]
            self.original_idx = 0
        elif name == 'shear':
            self.param_keys = ['shear', 'translate', 'angle', 'scale']
            a = torch.linspace(-160, 160, int(np.sqrt(self.k)) + 1)[:int(np.sqrt(self.k))].to(torch.float32).tolist()
            shear_params = [(a, b) for a, b in product(a, a)]
            self.param_vals = [
                shear_params,
                torch.zeros((self.k, 2)).tolist(),
                torch.zeros(self.k).tolist(),
                torch.ones(self.k).tolist()
            ]
            self.original_idx = shear_params.index((0.0, 0.0))
        elif name == 'resized_crop':
            self.param_keys = ['top', 'left', 'height', 'width', 'size']
            n = int(np.sqrt(np.sqrt(self.k)))
            a = (torch.linspace(0, 0.25, n) * resize).to(torch.float32).tolist()
            b = (torch.linspace(0.75, 0.25, n) * resize).to(torch.float32).tolist()
            p = product(a, a, b, b)
            a, b, c, d = tuple(zip(*p))
            self.param_vals = [
                a, b, c, d,
                [(s.item(), s.item()) for s in torch.ones(self.k, dtype=int) * crop_size]
            ]
            self.original_idx = 0
        elif name in ['h_flip', 'v_flip']:
            self.param_keys = ['aug']
            self.param_vals = [[False, True]]
            self.original_idx = 0
        elif name == 'deform':
            torch.manual_seed(0)
            np.random.seed(0)
            self.param_keys = ['points', 'sigma'] # 10, 50, 3, 9
            points = torch.repeat_interleave(torch.arange(2, 10), 32).tolist()
            sigma = torch.linspace(10, 50, 8).to(int).repeat(32).tolist()
            self.param_vals = [
                points,
                sigma
            ]
            self.original_idx = 0
        elif name == 'grayscale':
            self.param_keys = ['aug', 'num_output_channels']
            self.param_vals = [[False, True], [3, 3]]
            self.original_idx = 0
        elif name == 'brightness':
            self.param_keys = ['brightness_factor']
            self.param_vals = [torch.linspace(0.25, 5, self.k).to(torch.float32).tolist()]
            self.original_idx = self.k // 2
        elif name == 'contrast':
            self.param_keys = ['contrast_factor']
            self.param_vals = [torch.linspace(0.25, 5, self.k).to(torch.float32).tolist()]
            self.original_idx = self.k // 2
        elif name == 'saturation':
            self.param_keys = ['saturation_factor']
            self.param_vals = [torch.linspace(0.25, 5, self.k).to(torch.float32).tolist()]
            self.original_idx = self.k // 2
        elif name == 'hue':
            self.param_keys = ['hue_factor']
            self.param_vals = [torch.linspace(-0.5, 0.5, self.k).to(torch.float32).tolist()]
            self.original_idx = self.k // 2
        elif name == 'blur':
            self.param_keys = ['sigma', 'kernel_size']
            self.param_vals = [
                torch.linspace(1e-5, 20.0, self.k).to(torch.float32).tolist(),
                (torch.ones(self.k).to(int) + (crop_size // 20 * 2)).tolist(),
            ]
            self.original_idx = 0
        elif name == 'sharpness':
            self.param_keys = ['sharpness_factor']
            self.param_vals = [torch.linspace(1, 30.0, self.k).to(torch.float32).tolist()]
            self.original_idx = 0
        elif name in ['invert', 'equalize']:
            self.param_keys = ['aug']
            self.param_vals = [[False, True]]
            self.original_idx = 0
        elif name == 'posterize':
            self.param_keys = ['bits']
            self.param_vals = [torch.arange(1, 8).tolist()]
            self.original_idx = 0

    def T(self, image, **params):
        if 'aug' in params:
            if params['aug']:
                del params['aug']
                image = eval(f'FT.{self.fn}(image, **params)')
        elif self.fn == 'translation':
            pass
        else:
            image = eval(f'FT.{self.fn}(image, **params)')
        if self.fn != 'resized_crop':
            image = FT.resize(image, self.resize)
            if self.fn == 'translation':
                image = eval(f'FT.{self.fn}(image, **params)')
            image = FT.center_crop(image, self.crop_size)
        image = FT.pil_to_tensor(image).to(torch.float32)
        image = FT.normalize(image / 255., *self.norm)
        return image

    def __call__(self, x):
        xs = []
        for i in range(self.k):
            params = dict([(k, v[i]) for k, v in zip(self.param_keys, self.param_vals)])
            xs.append(self.T(x, **params))
        return tuple(xs)
