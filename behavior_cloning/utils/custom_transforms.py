from abc import ABC
import numpy as np
import cv2


class BaseTransform(ABC):
    def __init__(self):
        pass

    def __call__(self, x):
        return x

class ToFloat(BaseTransform):
    def __init__(self, pytorch=True):
        super(ToFloat, self).__init__()
        self.pytorch = pytorch

    def __call__(self, x):
        if self.pytorch:
            return x.astype(np.float32, copy=False)
        else:
            return float(x)

class DivideByScalar(BaseTransform):
    def __init__(self, scalar, axis=None, channels=None):
        super(DivideByScalar, self).__init__()
        self.scalar = scalar
        self.axis = axis
        self.channels = channels

    def __call__(self, x):
        if self.axis is not None and self.channels:
            # axis is dimension to index into
            # channels is list of channels on that axis to divide
            idx = [slice(None)] * x.ndim
            for c in self.channels:
                idx[self.axis] = c
                x[tuple(idx)] /= self.scalar
        else:
            x /= self.scalar

        return x


class NormalizeToRange(BaseTransform):
    def __init__(self, prev_min, prev_max, new_min, new_max):
        super(NormalizeToRange, self).__init__()
        self.prev_min = prev_min
        self.prev_max = prev_max
        self.new_min = new_min
        self.new_max = new_max

    def __call__(self, x):
        # return (x - self.prev_min) / (self.prev_max - self.prev_min) * (self.new_max - self.new_min) + self.new_min
        # same as above but using inplace operations
        x -= self.prev_min
        x /= (self.prev_max - self.prev_min)
        x *= (self.new_max - self.new_min)
        x += self.new_min
        return x


class ResizeImage(BaseTransform):
    def __init__(self, shape=None, mode=None):
        super(ResizeImage, self).__init__()
        self.shape = shape
        self.mode = mode

    def __call__(self, x):
        if self.mode == 'half':
            return cv2.resize(x, (x.shape[1]//2, x.shape[0]//2))
        return cv2.resize(x, self.shape)


class Reshape(BaseTransform):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def __call__(self, x):
        return x.reshape(self.shape)
    

class MoveAxis(BaseTransform):
    def __init__(self, orig_axis, new_axis):
        super(MoveAxis, self).__init__()
        self.orig_axis = orig_axis
        self.new_axis = new_axis

    def __call__(self, x):
        return np.moveaxis(x, self.orig_axis, self.new_axis)


class Rescale(BaseTransform):
    def __init__(self, scale):
        super(Rescale, self).__init__()
        self.scale = scale

    def __call__(self, x):
        return np.array([cv2.resize(frame, None, fx=self.scale, fy=self.scale) for frame in x])
    

class ClampToMax(BaseTransform):
    def __init__(self, max_val):
        super(ClampToMax, self).__init__()
        self.max_val = max_val

    def __call__(self, x):
        return np.clip(x, x.min(), self.max_val, out=x)


class VerticalFlip(BaseTransform):
    def __init__(self, p=0.0):
        super(VerticalFlip, self).__init__()
        self.p = p

    def __call__(self, x):
        if np.random.rand() < self.p:
            for img_type, img in x['imgs'].items():
                x['imgs'][img_type] = np.flip(img, 0).copy()
            if 'metadata' in x:
                x['metadata']['target_y_percent'] *= -1
        return x

    
class HorizontalFlip(BaseTransform):
    def __init__(self, p=0.0):
        super(HorizontalFlip, self).__init__()
        self.p = p

    def __call__(self, x):
        if np.random.rand() < self.p:
            for img_type, img in x['imgs'].items():
                x['imgs'][img_type] = np.flip(img, 1).copy() # need to copy to avoid negative strides
            if 'right_eye' in x['imgs'] and 'left_eye' in x['imgs']:
                x['imgs']['right_eye'], x['imgs']['left_eye'] = x['imgs']['left_eye'], x['imgs']['right_eye']
            if 'metadata' in x:
                x['metadata']['target_x_percent'] *= -1
        return x
    

class ColorJitter(BaseTransform):
    def __init__(self, p=0.0):
        super(ColorJitter, self).__init__()
        self.p = p

    def __call__(self, x):
        if np.random.rand() < self.p:
            random_hue_factor = np.random.uniform(0.95, 1.05)
            random_saturation_factor = np.random.uniform(0.5, 1.5)
            random_value_factor = np.random.uniform(0.4, 1.6)

            img = x

            img = img.astype(np.float32, copy=False)
            # put channels last
            # img = np.moveaxis(img, 0, -1)

            img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2HSV)
            # jitter hue saturation and value by multiplying by random values
            img[:, :, 0] *= random_hue_factor
            img[:, :, 1] *= random_saturation_factor
            img[:, :, 2] *= random_value_factor
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
            # clip
            img = np.clip(img, 0, 255, out=img)

            # show original and jittered image side by side
            # import matplotlib.pyplot as plt

            # fig, ax = plt.subplots(1, 2)
            # ax[0].imshow(x[:, :, :3].astype(np.uint8))
            # ax[1].imshow(img.astype(np.uint8))
            # plt.show()


            x[:, :, :3] = img.astype(np.uint8, copy=False)
        return x


class PoissonNoise(BaseTransform):
    # additive noise to first 3 channels
    def __init__(self, p=0.0, mean=2):
        super(PoissonNoise, self).__init__()
        self.p = p
        self.mean = mean

    def __call__(self, x):
        if np.random.rand() < self.p:
            noise_map = np.random.poisson(self.mean, x.shape[:-1]).astype(type(x[0, 0, 0]))
            x[:, :, :3] += noise_map[..., np.newaxis]
            x[:, :, :3] = np.clip(x[:, :, :3], 0, 255, out=x[:, :, :3])

        return x
    
    
class ImageShift(BaseTransform):
    def __init__(self, p=0.0, padding=5):
        super(ImageShift, self).__init__()
        self.p = p
        self.padding = padding

    def __call__(self, x):
        if np.random.rand() < self.p:
            # pad with average value for each channel
            x = np.pad(x, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='mean')
            shift_x = np.random.randint(-self.padding, self.padding)
            shift_y = np.random.randint(-self.padding, self.padding)
            x = x[self.padding + shift_y: -self.padding + shift_y, self.padding + shift_x: -self.padding + shift_x]
        return x


class ImageScale(BaseTransform):
    def __init__(self, p=0.0, max_scale=0.05):
        super(ImageScale, self).__init__()
        self.p = p
        self.max_scale = max_scale

    def __call__(self, x):
        if np.random.rand() < self.p:
            scale = 1 + np.random.uniform(-self.max_scale, self.max_scale)
            orig_size = x.shape[:2]
            x = cv2.resize(x, None, fx=scale, fy=scale)
            # center crop
            x = x[(x.shape[0] - orig_size[0]) // 2: (x.shape[0] + orig_size[0]) // 2,
                  (x.shape[1] - orig_size[1]) // 2: (x.shape[1] + orig_size[1]) // 2]
        return x

    
class RandomCrop(BaseTransform):
    def __init__(self):
        super(RandomCrop, self).__init__()

    def __call__(self, x):
        # random crop right and left eye images by 4 pixels smaller
        for img_type, img in x['imgs'].items():
            if 'eye' in img_type:
                x1 = np.random.randint(4)
                y1 = np.random.randint(4)
                x2 = x1 + 124
                y2 = y1 + 124
                x['imgs'][img_type] = img[:, y1:y2, x1:x2]
        return x
    

class RandomJitter(BaseTransform):
    def __init__(self, p=0.0, jitter=0.1):
        super(RandomJitter, self).__init__()
        self.p = p
        self.jitter = jitter

    def __call__(self, x):
        if np.random.rand() < self.p:
            x += np.random.uniform(-self.jitter, self.jitter, size=x.shape)
        return x