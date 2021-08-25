from networks.ours import UNet, DualStreamUNet


def load_network(cfg):
    architecture = cfg.MODEL.TYPE
    if architecture == 'unet':
        return UNet(cfg)
    elif architecture == 'dualstreamunet':
        return DualStreamUNet(cfg)
    else:
        return UNet(cfg)
