import cv2
from skimage import io
import numpy as np
from pathlib import Path


def stretch_8bit(band, lower_percent=2, higher_percent=98):
    a = 0
    b = 255
    real_values = band.flatten()
    real_values = real_values[real_values > 0]
    c = np.percentile(real_values, lower_percent)
    d = np.percentile(real_values, higher_percent)
    t = a + (band - c) * (b - a) / float(d - c)
    t[t < a] = a
    t[t > b] = b
    return t.astype(np.uint8) / 255.


def histogram_match(source, reference, match_proportion=1.0):
    orig_shape = source.shape
    source = source.ravel()

    if np.ma.is_masked(reference):
        reference = reference.compressed()
    else:
        reference = reference.ravel()

    s_values, s_idx, s_counts = np.unique(
        source, return_inverse=True, return_counts=True)
    r_values, r_counts = np.unique(reference, return_counts=True)
    s_size = source.size

    if np.ma.is_masked(source):
        mask_index = np.ma.where(s_values.mask)
        s_size = np.ma.where(s_idx != mask_index[0])[0].size
        s_values = s_values.compressed()
        s_counts = np.delete(s_counts, mask_index)

    s_quantiles = np.cumsum(s_counts).astype(np.float64) / s_size
    r_quantiles = np.cumsum(r_counts).astype(np.float64) / reference.size

    interp_r_values = np.interp(s_quantiles, r_quantiles, r_values)

    if np.ma.is_masked(source):
        interp_r_values = np.insert(interp_r_values, mask_index[0], source.fill_value)

    target = interp_r_values[s_idx]

    if match_proportion is not None and match_proportion != 1:
        diff = source - target
        target = source - (diff * match_proportion)

    if np.ma.is_masked(source):
        target = np.ma.masked_where(s_idx == mask_index[0], target)
        target.fill_value = source.fill_value

    return target.reshape(orig_shape)


if __name__ == '__main__':

    FOLDER = Path('C:/Users/hafne/urban_change_detection/data/Onera/')

    IMG_FOLDER = FOLDER / 'images'
    # folder of the form ./IMGS_PREPROCESSED/abudhabi/imgs_1/..(13 tif 2D images of sentinel channels)..
    # ./IMGS_PREPROCESSED/abudhabi/imgs_2/..(13 tif 2D images of sentinel channels)..
    # ....
    # ./IMGS_PREPROCESSED/abudhabi/imgs_n/..(13 tif 2D images of sentinel channels)..
    # where n = number of dates

    # here you specify which dates you want to use
    nb_dates = [1, 2]
    channels = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']

    # all areas of the OSCD dataset
    all_areas = ['abudhabi', 'aguasclaras', 'beihai', 'beirut', 'bercy', 'bordeaux', 'brasilia', 'chongqing',
                 'cupertino', 'dubai', 'hongkong', 'lasvegas', 'milano', 'montpellier', 'mumbai', 'nantes',
                 'norcia', 'paris', 'pisa', 'rennes', 'rio', 'saclay_e', 'saclay_w', 'valencia']

    DESTINATION_FOLDER = FOLDER / 'images_preprocessed'
    if not DESTINATION_FOLDER.exists():
        DESTINATION_FOLDER.mkdir()

    for i_path in all_areas:
        print(i_path)

        date_folders = []
        for nd in nb_dates:
            temp_path = IMG_FOLDER / i_path / f'imgs_{nd}'
            files = [file for file in temp_path.glob('**/*')]
            date_folders.append(files)

        # B02 channel has the same dimensions with the groundtruth for the labeled images.
        # So we keep it to reshape the rest of the channels for both labeled images and nonlabeled images
        gts = [s for s in date_folders[0] if 'B02' in str(s)]
        gts = io.imread(gts[0])

        temp_path = DESTINATION_FOLDER / i_path
        if not temp_path.exists():
            temp_path.mkdir()

        for nd in nb_dates:
            print('date', nd)
            imgs = []
            if nd == 1:
                for ch in channels:
                    im = [s for s in date_folders[nd-1] if ch in str(s)]
                    im = io.imread(im[0])
                    im[im > 5500] = 5500
                    im = stretch_8bit(im)
                    im = cv2.resize(im, (gts.shape[1], gts.shape[0]))
                    im = np.reshape(im, (gts.shape[0], gts.shape[1], 1))
                    imgs.append(im)
                imgs0 = imgs
            else:

                for ch in channels:
                    im = [s for s in date_folders[nd-1] if ch in str(s)]
                    im = io.imread(im[0])
                    im[im > 5500] = 5500
                    im = stretch_8bit(im)
                    im = histogram_match(im, imgs0[channels.index(ch)])
                    im = cv2.resize(im, (gts.shape[1], gts.shape[0]))
                    im = np.reshape(im, (gts.shape[0], gts.shape[1], 1))
                    imgs.append(im)

            im_merge = np.stack(imgs, axis=2)
            im_merge = np.asarray(im_merge)
            im_merge = np.reshape(im_merge, (im_merge.shape[0], im_merge.shape[1], im_merge.shape[2]))

            im_file = DESTINATION_FOLDER / i_path / f'{i_path}_{nd}.npy'
            np.save(str(im_file), im_merge)
