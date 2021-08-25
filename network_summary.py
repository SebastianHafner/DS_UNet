import torch
from torchsummary import summary
from networks.network_loader import load_network
from experiment_manager.config import new_config
from pathlib import Path
CFG_DIR = Path.cwd() / 'configs'


# loading cfg for inference
def load_cfg(cfg_file: Path):
    cfg = new_config()
    cfg.merge_from_file(str(cfg_file))
    return cfg


if __name__ == '__main__':

    cfg = 'fusion_dualstream_1'

    cfg_file = CFG_DIR / f'{cfg}.yaml'

    # loading cfg and network
    cfg = load_cfg(cfg_file)

    net = load_network(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # TODO: replace by number of bands function
    h = w = cfg.AUGMENTATION.CROP_SIZE
    img_channels = cfg.MODEL.IN_CHANNELS // 2
    img_size = (img_channels, h, w)

    summary(net, input_size=[img_size, img_size])
