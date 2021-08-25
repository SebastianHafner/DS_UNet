import argparse

def default_argument_parser():
    """
    Create a parser with some common arguments used by detectron2 users.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="Experiment Args")
    parser.add_argument('-c',"--config-file", dest='config_file', default="", required=True, metavar="FILE", help="path to config file")
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
