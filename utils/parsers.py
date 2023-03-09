import argparse


def training_argument_parser():
    parser = argparse.ArgumentParser(description="Experiment Args")
    parser.add_argument('-c', "--config-file", dest='config_file', required=True, metavar="FILE",
                        help="path to config file")
    parser.add_argument('-d', '--data-dir', dest='data_dir', required=True, type=str, help='dataset directory')
    parser.add_argument('-o', '--output-dir', dest='log_dir', required=True, type=str, help='output directory')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def preprocessing_argument_parser():
    parser = argparse.ArgumentParser(description="Experiment Args")
    parser.add_argument('-d', '--data-dir', dest='data_dir', required=True, type=str,
                        help='OSCD dataset directory')
    parser.add_argument('-p', '--preprocssed-dir', dest='preprocessed_dir', required=True, type=str,
                        help='Preprocessed data directory')
    parser.add_argument('-s', "--s1-dir", dest='s1_dir', default="data", required=False, metavar="FILE",
                        help="Sentinel-1 data directory")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def evaluation_argument_parser():
    parser = argparse.ArgumentParser(description="Experiment Args")
    parser.add_argument('-c', "--config-file", dest='config_file', required=True, metavar="FILE",
                        help="path to config file")
    parser.add_argument('-d', '--data-dir', dest='data_dir', required=True, type=str, help='dataset directory')
    parser.add_argument('-o', '--output-dir', dest='log_dir', required=True, type=str, help='output directory')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser
