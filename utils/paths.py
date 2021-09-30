import yaml
from pathlib import Path
from utils.experiment_manager import CfgNode

# set the paths
HOME_ROOT = '/home/shafner/DS_UNet'
OSCD_ROOT = '/storage/shafner/oscd_dataset'
SENTINEL1_DATA = '/home/shafner/DS_UNet/data'
PREPROCESSED_ROOT = '/storage/shafner/oscd_dataset_preprocessed'
OUTPUT_ROOT = '/storage/shafner/urban_change_detection/'


def load_paths() -> dict:
    C = CfgNode()
    C.HOME_ROOT = HOME_ROOT
    C.OSCD_ROOT = OSCD_ROOT
    C.SENTINEL1_DATA = SENTINEL1_DATA
    C.PREPROCESSED_ROOT = PREPROCESSED_ROOT
    C.OUTPUT_ROOT = OUTPUT_ROOT

    return C.clone()
