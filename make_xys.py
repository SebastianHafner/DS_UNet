import numpy as np
from skimage import io
from skimage.transform import rotate, resize
import os
import cv2
import pandas as pd
from pathlib import Path


def shuffle(vector):
    vector = np.asarray(vector)
    p = np.random.permutation(len(vector))
    vector = vector[p]
    return vector


def sliding_window_train(i_city, labeled_areas, label, window_size, step):
    city = []
    fpatches_labels = []

    x = 0
    while x != label.shape[0]:
        y = 0
        while y != label.shape[1]:

            if (not y + window_size > label.shape[1]) and (not x + window_size > label.shape[0]):
                line = np.array([x, y, labeled_areas.index(i_city), 0])
                # (x,y) are the saved coordinates,
                # labeled_areas.index(i_city)... are the image ids, e.g according to train_areas,
                # the indice for abudhabi in the list is 0, for beihai it is 1, for beirut is 3, etc..
                # the fourth element which has been set as 0, represents the transformadion index,
                # which in this case indicates that no data augmentation will be performed for the
                # specific patch

                city.append(line)

                # CONDITIONS
                new_patch_label = label[x:x + window_size, y:y + window_size]
                ff = np.where(new_patch_label == 2)
                perc = ff[0].shape[0] / float(window_size * window_size)
                if ff[0].shape[0] == 0:
                    stride = window_size
                else:
                    stride = step
                if perc >= 0.05:
                    # if percentage of change exceeds a threshold, perform data augmentation on this patch
                    # Below, 1, 2, 3 are transformation indexes that will be used by the custom dataloader
                    # to perform various rotations
                    line = np.array([x, y, labeled_areas.index(i_city), 1])
                    city.append(line)
                    line = np.array([x, y, labeled_areas.index(i_city), 2])
                    city.append(line)
                    line=np.array([x, y, labeled_areas.index(i_city), 3])
                    city.append(line)
                 # CONDITIONS

            if y + window_size == label.shape[1]:
                break

            if y + window_size > label.shape[1]:
                y = label.shape[1] - window_size
            else:
                y = y + stride

        if x + window_size == label.shape[0]:
            break

        if x + window_size > label.shape[0]:
            x = label.shape[0] - window_size
        else:
            x = x + stride

    return np.asarray(city)


if __name__ == '__main__':

    train_areas = ['abudhabi', 'beihai', 'aguasclaras', 'beirut', 'bercy', 'bordeaux', 'cupertino',
                   'hongkong', 'mumbai', 'nantes', 'rennes', 'saclay_e', 'pisa', 'rennes']

    FOLDER = Path('C:/Users/hafne/urban_change_detection/data/Onera/')

    step = 6
    patch_s = 32

    cities = []
    for i_city in train_areas:
        file = FOLDER / 'labels' / i_city / 'cm' / f'{i_city}-cm.tif'
        print('icity', i_city)
        train_gt = io.imread(file)
        xy_city = sliding_window_train(i_city, train_areas, train_gt, patch_s, step)
        cities.append(xy_city)

    # from all training (x,y) locations, divide 4/5 for training and 1/5 for validation
    final_cities = np.concatenate(cities, axis=0)
    size_len = len(final_cities)
    portion = int(size_len / 5)
    final_cities = shuffle(final_cities)
    final_cities_train = final_cities[:-portion]
    final_cities_val = final_cities[-portion:]

    # save train to csv file
    df = pd.DataFrame({'X': list(final_cities_train[:, 0]),
                       'Y': list(final_cities_train[:, 1]),
                       'image_ID': list(final_cities_train[:, 2]),
                       'transform_ID': list(final_cities_train[:, 3]),
                       })
    train_file = FOLDER / 'myxys_train.csv'
    df.to_csv(str(train_file), index=False, columns=["X", "Y", "image_ID", "transform_ID"])

    # save val to csv file
    df = pd.DataFrame({'X': list(final_cities_val[:, 0]),
                       'Y': list(final_cities_val[:, 1]),
                       'image_ID': list(final_cities_val[:, 2]),
                       'transform_ID': list(final_cities_val[:, 3]),
                       })
    val_file = FOLDER / 'myxys_val.csv'
    df.to_csv(str(val_file), index=False, columns=["X", "Y", "image_ID", "transform_ID"])
