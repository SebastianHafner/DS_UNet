import torch
from torch.utils import data as torch_data
from torchvision import transforms
from pathlib import Path
import numpy as np
import augmentations as aug
from utils import *

ORBITS = {
    'aguasclaras': [24],
    'bercy': [59, 8, 110],
    'bordeaux': [30, 8, 81],
    'nantes': [30, 81],
    'paris': [59, 8, 110],
    'rennes': [30, 81],
    'saclay_e': [59, 8],
    'abudhabi': [130],
    'cupertino': [35, 115, 42],
    'pisa': [15, 168],
    'beihai': [157],
    'hongkong': [11, 113],
    'beirut': [14, 87],
    'mumbai': [34],
    'brasilia': [24],
    'montpellier': [59, 37],
    'norcia': [117, 44, 22, 95],
    'rio': [155],
    'saclay_w': [59, 8, 110],
    'valencia': [30, 103, 8, 110],
    'dubai': [130, 166],
    'lasvegas': [166, 173],
    'milano': [66, 168],
    'chongqing': [55, 164]
}

SENTINEL1_BANDS = ['VV']
SENTINEL2_BANDS = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']


class OSCDDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, dataset: str, no_augmentation: bool = False):
        super().__init__()

        self.cfg = cfg
        self.root_dir = Path(cfg.DATASET.PATH)

        if dataset == 'train':
            multiplier = cfg.DATASET.TRAIN_MULTIPLIER
            self.cities = multiplier * cfg.DATASET.TRAIN
        else:
            self.cities = cfg.DATASET.TEST

        self.length = len(self.cities)

        if no_augmentation:
            self.transform = transforms.Compose([aug.Numpy2Torch()])
        else:
            self.transform = aug.compose_transformations(cfg)

        self.mode = cfg.DATASET.MODE

        # creating boolean feature vector to subset sentinel 1 and sentinel 2 bands
        self.s1_band_selection = self._get_band_selection(SENTINEL1_BANDS, cfg.DATASET.SENTINEL1_BANDS)
        self.s2_band_selection = self._get_band_selection(SENTINEL2_BANDS, cfg.DATASET.SENTINEL2_BANDS)

    def __getitem__(self, index):

        city = self.cities[index]

        # randomly choosing an orbit for sentinel1
        orbit = np.random.choice(ORBITS[city])
        # orbit = ORBITS[city][0]

        if self.cfg.DATASET.MODE == 'optical':
            t1_img = self._get_sentinel2_data(city, 't1')
            t2_img = self._get_sentinel2_data(city, 't2')
        elif self.cfg.DATASET.MODE == 'sar':
            t1_img = self._get_sentinel1_data(city, orbit, 't1')
            t2_img = self._get_sentinel1_data(city, orbit, 't2')
        else:
            s1_t1_img = self._get_sentinel1_data(city, orbit, 't1')
            s2_t1_img = self._get_sentinel2_data(city, 't1')
            t1_img = np.concatenate((s1_t1_img, s2_t1_img), axis=2)

            s1_t2_img = self._get_sentinel1_data(city, orbit, 't2')
            s2_t2_img = self._get_sentinel2_data(city, 't2')
            t2_img = np.concatenate((s1_t2_img, s2_t2_img), axis=2)

        label = self._get_label_data(city)
        t1_img, t2_img, label = self.transform((t1_img, t2_img, label))

        sample = {
            't1_img': t1_img,
            't2_img': t2_img,
            'label': label,
            'city': city
        }

        return sample

    def _get_sentinel1_data(self, city, orbit, t):
        file = self.root_dir / city / 'sentinel1' / f'sentinel1_{city}_{orbit}_{t}.npy'
        img = np.load(file)[:, :, self.s1_band_selection]
        return img.astype(np.float32)

    def _get_sentinel2_data(self, city, t):
        file = self.root_dir / city / 'sentinel2' / f'sentinel2_{city}_{t}.npy'
        img = np.load(file)[:, :, self.s2_band_selection]
        return img.astype(np.float32)

    def _get_label_data(self, city):
        label_file = self.root_dir / city / 'label' / f'urbanchange_{city}.npy'
        label = np.load(label_file).astype(np.float32)
        label = label[:, :, np.newaxis]
        return label

    @ staticmethod
    def _get_band_selection(features, selection):
        feature_selection = [False for _ in range(len(features))]
        for feature in selection:
            i = features.index(feature)
            feature_selection[i] = True
        return feature_selection

    def __len__(self):
        return self.length

    def band_index(self, band: str) -> int:
        s1_bands = self.cfg.DATASET.SENTINEL1_BANDS
        s2_bands = self.cfg.DATASET.SENTINEL2_BANDS
        mode = self.cfg.DATASET.MODE

        index = s1_bands.index(band) if band in s1_bands else s2_bands.index(band)

        # handle band concatenation for fusion
        if mode == 'fusion' and band in s2_bands:
            index += len(s1_bands)

        return index

    def sampler(self):
        if self.cfg.AUGMENTATION.OVERSAMPLING == 'pixel':
            sampling_weights = np.array([float(self._get_label_data(city).size) for city in self.cities])
        if self.cfg.AUGMENTATION.OVERSAMPLING == 'change':
            sampling_weights = np.array([float(np.sum(self._get_label_data(city))) for city in self.cities])
        sampler = torch_data.WeightedRandomSampler(weights=sampling_weights, num_samples=self.length,
                                                   replacement=True)
        return sampler


class OSCDDifferenceImages(OSCDDataset):

    def __init__(self, cfg, dataset: str, no_augmentation: bool = False):
        super().__init__(cfg, dataset, no_augmentation)

    def __getitem__(self, index):

        city = self.cities[index]

        # randomly choosing an orbit for sentinel1
        orbit = np.random.choice(ORBITS[city])

        if self.mode == 'optical':
            t1_img = self._get_sentinel2_data(city, 't1')
            t2_img = self._get_sentinel2_data(city, 't2')
            diff_img = self.optical_difference_image(t1_img, t2_img)
        elif self.mode == 'sar':
            t1_img = self._get_sentinel1_data(city, orbit, 't1')
            t2_img = self._get_sentinel1_data(city, orbit, 't2')
            diff_img = self.sar_difference_image(t1_img, t2_img)
        else:
            s1_t1_img = self._get_sentinel1_data(city, orbit, 't1')
            s1_t2_img = self._get_sentinel1_data(city, orbit, 't2')
            s1_diff_img = self.sar_difference_image(s1_t1_img, s1_t2_img)

            s2_t1_img = self._get_sentinel2_data(city, 't1')
            s2_t2_img = self._get_sentinel2_data(city, 't2')
            s2_diff_img = self.optical_differece_image(s2_t1_img, s2_t2_img)

            diff_img = np.concatenate((s1_diff_img, s2_diff_img), axis=2)

        label = self._get_label_data(city)
        diff_img, _, label = self.transform((diff_img, diff_img, label))

        sample = {
            'diff_img': diff_img,
            'label': label,
            'city': city
        }

        return sample

    def optical_difference_image(self, t1_img: np.ndarray, t2_img: np.ndarray) -> np.ndarray:

        if self.cfg.DATASET.INDICES:
            # change vector analysis with ndvi and ndbi
            red = self.band_index('B04')
            nir = self.band_index('B08')
            swir = self.band_index('B11')

            t1_ndvi = normalize(normalized_difference(t1_img, red, nir), -1, 1)
            t1_ndbi = normalize(normalized_difference(t1_img, swir, nir), -1, 1)
            t1_indices = np.stack((t1_ndvi, t1_ndbi), axis=-1)

            t2_ndvi = normalize(normalized_difference(t2_img, red, nir), -1, 1)
            t2_ndbi = normalize(normalized_difference(t2_img, swir, nir), -1, 1)
            t2_indices = np.stack((t2_ndvi, t2_ndbi), axis=-1)
            t1_img, t2_img = t1_indices, t2_indices

        cva = change_vector_analysis(t1_img, t2_img)

        return cva.astype(np.float32)

    def sar_difference_image(self, t1_img: np.ndarray, t2_img: np.ndarray) -> np.ndarray:

        # log ration of VV
        vv = self.band_index('VV')

        t1_vv = t1_img[:, :, vv]
        t2_vv = t2_img[:, :, vv]

        lr = log_ratio(t1_vv, t2_vv)
        # TODO: add normalization
        return lr.astype(np.float32)
