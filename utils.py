import torch
import rasterio
from pathlib import Path
import math
import numpy as np

EPSILON = 1e-10


# reading in geotiff file as numpy array
def read_tif(file: Path):

    if not file.exists():
        raise FileNotFoundError(f'File {file} not found')

    with rasterio.open(file) as dataset:
        arr = dataset.read()  # (bands X height X width)
        transform = dataset.transform
        crs = dataset.crs

    return arr.transpose((1, 2, 0)), transform, crs


# writing an array to a geo tiff file
def write_tif(file: Path, arr, transform, crs):

    if not file.parent.exists():
        file.parent.mkdir()

    height, width, bands = arr.shape
    with rasterio.open(
            file,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=bands,
            dtype=arr.dtype,
            crs=crs,
            transform=transform,
    ) as dst:
        for i in range(bands):
            dst.write(arr[:, :, i], i + 1)


def to_numpy(tensor:torch.Tensor):
    return tensor.cpu().detach().numpy()

# returns 2d distance array from two n-dimensional arrays
def euclidean_distance(t1_img: np.ndarray, t2_img: np.ndarray) -> np.ndarray:
    diff = t1_img - t2_img
    diff_squared = np.square(diff)
    distance = np.sqrt(np.sum(diff_squared, axis=-1))
    return distance


# according to: https://www.geo.uzh.ch/microsite/rsl-documents/research/publications/other-sci-communications/2009_SAMAnisotropy_SPIE_JW-0471040512/2009_SAMAnisotropy_SPIE_JW.pdf
def spectral_angle_mapper(t1_img: np.ndarray, t2_img: np.ndarray, radians: bool = True) -> np.ndarray:

    nominator = np.sum(t1_img * t2_img, axis=-1)

    denominator1 = np.sqrt(np.sum(np.square(t1_img), axis=-1))
    denominator2 = np.sqrt(np.sum(np.square(t2_img), axis=-1))
    denominator = denominator1 * denominator2 + EPSILON

    a_radians = np.arccos((nominator / denominator))

    return a_radians if radians else a_radians * 180 / math.pi


# references:
# https://reader.elsevier.com/reader/sd/pii/S1878029611008607?token=95EB0C4C341A3C82861FDB0F250C9315C15C62CF192A4A3823750BB21E22DBDCC17D19E07E019CF03FCA328AAEAA3718
# http://remote-sensing.org/change-vector-analysis-explained-graphically/
# http://www.gitta.info/ThemChangeAna/en/html/TimeChAn_learningObject1.html
def change_vector_analysis(t1_img: np.ndarray, t2_img: np.ndarray) -> np.ndarray:
    # images with shape (H x W x 2)
    shape = t1_img.shape
    n = shape[-1]

    # computing magnitude
    magnitude = euclidean_distance(t1_img, t2_img)
    # magnitude = normalize(magnitude, 0, 1.414214)
    magnitude = magnitude[:, :, None]

    # computing n - 1 angles (1 for bivariate)
    change_vectors = t2_img - t1_img
    reference_vector = np.zeros((shape[0], shape[1], 2))
    reference_vector[:, :, 1] = 1
    directions = np.zeros((shape[0], shape[1], n - 1))
    for i in range(n - 1):
        change_vector = change_vectors[:, :, [i, -1]]
        direction = spectral_angle_mapper(change_vector, reference_vector)
        directions[:, :, 0] = direction
        # directions[:, :, 0] = normalize(direction, 0, math.pi)

    cva = np.concatenate((magnitude, directions), axis=-1)

    return cva


def log_ratio(t1_img: np.ndarray, t2_img: np.ndarray) -> np.ndarray:
    logR = t1_img - t2_img
    logR = normalize(logR, -1, 1)
    return logR


def normalize(img: np.ndarray, from_min: float, from_max: float) -> np.ndarray:
    return (img - from_min) / (from_max - from_min + EPSILON)


def normalized_difference(img: np.ndarray, band1: int, band2: int):
    img1, img2 = img[:, :, band1], img[:, :, band2]
    return (img1 - img2) / (img1 + img2 + EPSILON)


if __name__ == '__main__':

    # a = torch.ones(2, 4, 4).to('cuda')
    # a[0, ] = 3
    # a[1, ] = 4
    # b = torch.ones(2, 4, 4).to('cuda')
    # b[0, ] = 4
    # b[1, ] = 3
    # distance = euclidean_distance(a, b)
    # print(distance)
    #
    # direction = spectral_angle_mapper(a, b)
    # print(direction)

    a = np.zeros(4, 4, 2)
    b = np.ones(4, 4, 2)
    b[:, :, 0] = 0
    change_vector_analysis(a, b)

    pass
