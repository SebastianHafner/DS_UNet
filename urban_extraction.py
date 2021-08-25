import torch
from torch.utils import data as torch_data
import numpy as np

from networks.ours import UNet
from experiment_manager.config import new_config
import datasets

import matplotlib.pyplot as plt

from pathlib import Path


# loading cfg for inference
def load_cfg(cfg_file: Path):
    cfg = new_config()
    cfg.merge_from_file(str(cfg_file))
    return cfg


# loading network for inference
def load_net(cfg, net_file: Path, device):
    net = UNet(cfg)
    state_dict = torch.load(str(net_file), map_location=lambda storage, loc: storage)
    net.load_state_dict(state_dict)
    return net.to(device)


def classify(img, net, threshold, return_numpy=True):
    y_logits = net(img)
    y_prob = torch.sigmoid(y_logits)
    y_pred = y_prob > threshold
    if return_numpy:
        return torch2numpy(y_pred, 'uint8'), torch2numpy(y_prob)
    return y_pred, y_prob


def torch2numpy(tensor: torch.tensor, nptype: str = 'float32'):
    cpu_tensor = tensor.cpu().detach()
    arr = cpu_tensor.numpy().astype(nptype)
    if len(arr.shape) == 4:
        transpose = (0, 2, 3, 1)
    elif len(arr.shape) == 3:
        transpose = (1, 2, 0)
    else:
        transpose = (0, 1)
    arr = arr.transpose(transpose)
    return arr



def visual_evaluation(net_cfg_file: Path, net_file: Path, ds_cfg_file: Path, dataset: str = 'test',
                      save_path: Path = None):

    mode = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(mode)

    # loading network
    net_cfg = load_cfg(net_cfg_file)
    net = load_net(net_cfg, net_file, device)

    # loading dataset
    ds_cfg = load_cfg(ds_cfg_file)
    dataset = datasets.OSCDDataset(ds_cfg, dataset, no_augmentation=True)
    dataloader_kwargs = {
        'batch_size': 1,
        'num_workers': 0,
        'shuffle': False,
        'pin_memory': True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)
    threshold = net_cfg.THRESH

    with torch.no_grad():
        net.eval()
        for step, batch in enumerate(dataloader):

            fig, axs = plt.subplots(1, 4, figsize=(20, 10))

            city = batch['city'][0]
            print(city)

            t1_img = batch['t1_img'].to(device)
            t2_img = batch['t2_img'].to(device)
            y_true = batch['label'].to(device)

            data = {'pred': [], 'prob': [], 'rgb': []}
            for i, img in enumerate([t1_img, t2_img]):
                img_arr = torch2numpy(img)
                y_pred, y_prob = classify(img, net, threshold, return_numpy=True)
                data['pred'].append(y_pred[0, :, :, 0])
                data['prob'].append(y_prob[0, :, :, 0])

                img_arr = img_arr[0, ...]
                rgb = img_arr[:, :, [2, 1, 0]]
                rgb = np.minimum(rgb / 0.3, 1)
                data['rgb'].append(rgb)

                axs[i].imshow(y_prob[0, :, :, 0], vmin=0, vmax=1)
                # axs[i*2+1].imshow(y_prob[0, :, :, 0])

            label_arr = torch2numpy(y_true, 'uint8')
            axs[2].imshow(label_arr[0, :, :, 0])
            di = data['prob'][1]-data['prob'][0]
            axs[3].imshow(di, vmin=0, vmax=1)

            for ax in axs:
                ax.set_axis_off()

            assert(save_path.exists())
            file = save_path / f'urban_extraction_{city}.png'
            plt.savefig(file, dpi=300, bbox_inches='tight')
            plt.close()


if __name__ == '__main__':

    save_path = Path('/storage/shafner/urban_change_detection/urban_extraction')

    # network
    ue_cfg = 'baseline_sentinel2'
    ue_cfg_path = Path('/home/shafner/urban_dl/configs/urban_extraction')
    ue_cfg_file = ue_cfg_path / f'{ue_cfg}.yaml'
    ue_net_path = Path('/storage/shafner/run_logs/unet')
    ue_net_file = ue_net_path / ue_cfg / 'best_net.pkl'

    # cfg
    ds_cfg = 'urban_extraction_loader'
    ds_cfg_file = Path.cwd() / 'configs' / f'{ds_cfg}.yaml'

    visual_evaluation(ue_cfg_file, ue_net_file, ds_cfg_file, dataset='test', save_path=save_path)
