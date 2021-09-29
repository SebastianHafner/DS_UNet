import argparse
import logging
from tabulate import tabulate
from collections import OrderedDict
import yaml
from fvcore.common.config import CfgNode as _CfgNode


# from .config import CfgNode, new_config, global_config


def default_argument_parser():
    """
    Create a parser with some common arguments used by detectron2 users.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="Experiment Args")
    parser.add_argument('-c', "--config-file", dest='config_file', default="", required=True, metavar="FILE",
                        help="path to config file")
    parser.add_argument('-d', '--data-dir', dest='data_dir', type=str,
                        default='', help='dataset directory')
    parser.add_argument('-o', '--output-dir', dest='log_dir', type=str,
                        default='', help='output directory')
    parser.add_argument(
        "--resume",
        dest='resume',
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument('--resume-from', dest='resume_from', type=str,
                        default='', help='path of which the model will be loaded from')
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")

    # Hacky hack
    # parser.add_argument("--eval-training", action="store_true", help="perform evaluation on training set only")

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def new_config():

    C = CfgNode()

    C.CONFIG_DIR = 'config/'
    C.OUTPUT_BASE_DIR = 'output/'

    C.SEED = 7

    C.MODEL = CfgNode()
    C.MODEL.TYPE = 'unet'
    C.MODEL.OUT_CHANNELS = 1
    C.MODEL.IN_CHANNELS = 2
    C.MODEL.LOSS_TYPE = 'FrankensteinLoss'

    C.DATALOADER = CfgNode()
    C.DATALOADER.NUM_WORKER = 8
    C.DATALOADER.SHUFFLE = True

    C.DATASET = CfgNode()
    C.DATASET.PATH = ''
    C.DATASET.MODE = ''
    C.DATASET.SENTINEL1 = CfgNode()
    C.DATASET.SENTINEL1.BANDS = ['VV', 'VH']
    C.DATASET.SENTINEL1.TEMPORAL_MODE = 'bi-temporal'
    C.DATASET.SENTINEL2 = CfgNode()
    C.DATASET.SENTINEL2.BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
    C.DATASET.SENTINEL2.TEMPORAL_MODE = 'bi-temporal'
    C.DATASET.ALL_CITIES = []
    C.DATASET.TEST_CITIES = []

    C.OUTPUT_BASE_DIR = ''

    C.TRAINER = CfgNode()
    C.TRAINER.LR = 1e-4
    C.TRAINER.BATCH_SIZE = 16
    C.TRAINER.EPOCHS = 50

    C.AUGMENTATION = CfgNode()
    C.AUGMENTATION.CROP_TYPE = 'none'
    C.AUGMENTATION.CROP_SIZE = 32
    C.RANDOM_FLIP = True
    C.RANDOM_ROTATE = True
    return C.clone()


def setup(args):
    cfg = new_config()
    cfg.merge_from_file(f'configs/{args.config_file}.yaml')
    cfg.merge_from_list(args.opts)
    cfg.NAME = args.config_file
    return cfg


class CfgNode(_CfgNode):
    """
    The same as `fvcore.common.config.CfgNode`, but different in:

    1. Use unsafe yaml loading by default.
      Note that this may lead to arbitrary code execution: you must not
      load a config file from untrusted sources before manually inspecting
      the content of the file.
    2. Support config versioning.
      When attempting to merge an old config, it will convert the old config automatically.

    """

    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        # Always allow merging new configs
        self.__dict__[CfgNode.NEW_ALLOWED] = True
        super(CfgNode, self).__init__(init_dict, key_list, True)

    # Note that the default value of allow_unsafe is changed to True
    def merge_from_file(self, cfg_filename: str, allow_unsafe: bool = True) -> None:
        loaded_cfg = _CfgNode.load_yaml_with_base(cfg_filename, allow_unsafe=allow_unsafe)
        loaded_cfg = type(self)(loaded_cfg)

        # defaults.py needs to import CfgNode
        self.merge_from_other_cfg(loaded_cfg)
