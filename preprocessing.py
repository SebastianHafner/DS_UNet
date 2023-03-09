import numpy as np
from pathlib import Path
import cv2
import tifffile
from utils import parsers


def load_cities(dataset: str) -> list:
    dirs = paths.load_paths()
    cities_file = Path(dirs.OSCD_ROOT) / 'images' / f'{dataset}.txt'
    with open(cities_file, 'r') as f:
        cities = f.read()[:-1].split(',')
    return cities


CITIES = ['aguasclaras', 'bercy', 'bordeaux', 'nantes', 'paris', 'rennes', 'saclay_e', 'abudhabi', 'cupertino',
          'pisa', 'beihai', 'hongkong', 'beirut', 'mumbai', 'brasilia', 'montpellier', 'norcia', 'rio', 'saclay_w',
          'valencia', 'dubai', 'lasvegas', 'milano', 'chongqing']


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


def get_band(file: Path) -> str:
    return file.stem.split('_')[-1]


def combine_bands(folder: Path) -> np.ndarray:

    bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
    n_bands = len(bands)

    # using blue band as reference (10 m) to create img
    blue_file = folder / 'B02.tif'
    blue = tifffile.imread(str(blue_file))
    img = np.ndarray((*blue.shape, n_bands), dtype=np.float32)

    for i, band in enumerate(bands):
        band_file = folder / f'{band}.tif'
        arr = tifffile.imread(str(band_file))
        band_h, band_w = arr.shape

        # up-sample 20 m bands
        # arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_CUBIC)

        # rescaling image to [0, 1]
        arr = np.clip(arr / 10000, a_min=0, a_max=1)
        img[:, :, i] = arr

    return img


def process_city(city: str, oscd_root: str, preprocessed_root: str):

    print(f'Preprocessing {city}')

    dirs = paths.load_paths()

    new_parent = Path(preprocessed_root) / city
    new_parent.mkdir(exist_ok=True)

    # image data
    for t in [1, 2]:

        # get data
        from_folder = Path(oscd_root) / 'images' / city / f'imgs_{t}_rect'
        img = combine_bands(from_folder)

        # save data
        to_folder = new_parent / 'sentinel2'
        to_folder.mkdir(exist_ok=True)

        save_file = to_folder / f'sentinel2_{city}_t{t}.npy'
        np.save(save_file, img)

    test_cities = load_cities('test')
    dataset = 'test' if city in test_cities else 'train'
    from_label_file = Path(oscd_root) / f'{dataset}_labels' / city / 'cm' / f'{city}-cm.tif'
    label = tifffile.imread(str(from_label_file))
    label = label - 1

    to_label_file = new_parent / 'label' / f'urbanchange_{city}.npy'
    to_label_file.parent.mkdir(exist_ok=True)
    np.save(to_label_file, label)


def add_sentinel1(city: str, orbit: int, oscd_root: str, preprocessed_root: str, sentinel1_data: str):

    dirs = paths.load_paths()

    test_cities = load_cities('test')
    dataset = 'test' if city in test_cities else 'train'
    label_file = Path(oscd_root) / f'{dataset}_labels' / city / 'cm' / f'{city}-cm.tif'
    label = tifffile.imread(str(label_file))
    h, w = label.shape

    for t in [1, 2]:

        s1_file =  Path(sentinel1_data) / f'sentinel1_{city}_{orbit}_t{t}.tif'
        img = tifffile.imread(str(s1_file))

        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        img = img[:, :, None]

        # save data
        to_folder = Path(preprocessed_root) / city / 'sentinel1'
        to_folder.mkdir(exist_ok=True)

        save_file = to_folder / f'sentinel1_{city}_{orbit}_t{t}.npy'
        np.save(save_file, img)


if __name__ == '__main__':
    args = parsers.preprocessing_argument_parser().parse_known_args()[0]
    for city in CITIES:
        process_city(city)
        orbits = ORBITS[city]
        for orbit in orbits:
            add_sentinel1(city, orbit)
