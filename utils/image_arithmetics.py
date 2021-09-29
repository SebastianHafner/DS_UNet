import torch
import math
import numpy as np

EPSILON = 1e-10


def to_numpy(tensor: torch.Tensor):
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
