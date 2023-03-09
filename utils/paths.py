import yaml
from pathlib import Path
from utils.experiment_manager import CfgNode

# set the paths
HOME_ROOT = 'C:/Users/shafner/DS_UNet'
OSCD_ROOT = 'C:/Users/shafner/datasets/OSCD_ROOT'
SENTINEL1_DATA = 'C:/Users/shafner/repos/DS_UNet/data'
PREPROCESSED_ROOT = 'C:/Users/shafner/datasets/OSCD_ROOT/preprocessed_root'
OUTPUT_ROOT = 'C:/Users/shafner/ds_unet'


def load_paths() -> dict:
    C = CfgNode()
    C.HOME_ROOT = HOME_ROOT
    C.OSCD_ROOT = OSCD_ROOT
    C.SENTINEL1_DATA = SENTINEL1_DATA
    C.PREPROCESSED_ROOT = PREPROCESSED_ROOT
    C.OUTPUT_ROOT = OUTPUT_ROOT

    return C.clone()
