import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np
import torch


def compose_transformations(cfg):
    transformations = []

    if cfg.AUGMENTATION.CROP_TYPE == 'uniform':
        transformations.append(UniformCrop(crop_size=cfg.AUGMENTATION.CROP_SIZE))
    elif cfg.AUGMENTATION.CROP_TYPE == 'importance':
        transformations.append(ImportanceRandomCrop(crop_size=cfg.AUGMENTATION.CROP_SIZE))

    if cfg.AUGMENTATION.RANDOM_FLIP:
        transformations.append(RandomFlip())

    if cfg.AUGMENTATION.RANDOM_ROTATE:
        transformations.append(RandomRotate())

    transformations.append(Numpy2Torch())

    return transforms.Compose(transformations)


class Numpy2Torch(object):
    def __call__(self, sample: tuple):
        img1, img2, label = sample
        img1_tensor = TF.to_tensor(img1)
        img2_tensor = TF.to_tensor(img2)
        label_tensor = TF.to_tensor(label)
        return img1_tensor, img2_tensor, label_tensor


class RandomFlip(object):
    def __call__(self, sample):
        img1, img2, label = sample
        h_flip = np.random.choice([True, False])
        v_flip = np.random.choice([True, False])

        if h_flip:
            img1 = np.flip(img1, axis=1).copy()
            img2 = np.flip(img2, axis=1).copy()
            label = np.flip(label, axis=1).copy()

        if v_flip:
            img1 = np.flip(img1, axis=0).copy()
            img2 = np.flip(img2, axis=0).copy()
            label = np.flip(label, axis=0).copy()

        return img1, img2, label


class RandomRotate(object):
    def __call__(self, args):
        img1, img2, label = args
        k = np.random.randint(1, 4)  # number of 90 degree rotations
        img1 = np.rot90(img1, k, axes=(0, 1)).copy()
        img2 = np.rot90(img2, k, axes=(0, 1)).copy()
        label = np.rot90(label, k, axes=(0, 1)).copy()
        return img1, img2, label


# Performs uniform cropping on images
class UniformCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def random_crop(self, img1: np.ndarray, img2: np.ndarray, label: np.ndarray):
        height, width, _ = label.shape
        crop_limit_x = width - self.crop_size
        crop_limit_y = height - self.crop_size
        x = np.random.randint(0, crop_limit_x)
        y = np.random.randint(0, crop_limit_y)

        img1_crop = img1[y:y+self.crop_size, x:x+self.crop_size, ]
        img2_crop = img2[y:y + self.crop_size, x:x + self.crop_size, ]
        label_crop = label[y:y+self.crop_size, x:x+self.crop_size, ]
        return img1_crop, img2_crop, label_crop

    def __call__(self, sample: tuple):
        img1, img2, label = sample
        img1, img2, label = self.random_crop(img1, img2, label)
        return img1, img2, label


class ImportanceRandomCrop(UniformCrop):
    def __call__(self, sample):
        img1, img2, label = sample

        sample_size = 20
        balancing_factor = 5

        random_crops = [self.random_crop(img1, img2, label) for _ in range(sample_size)]
        crop_weights = np.array([crop_label.sum() for _, _, crop_label in random_crops]) + balancing_factor
        crop_weights = crop_weights / crop_weights.sum()

        sample_idx = np.random.choice(sample_size, p=crop_weights)
        img1, img2, label = random_crops[sample_idx]

        return img1, img2, label

