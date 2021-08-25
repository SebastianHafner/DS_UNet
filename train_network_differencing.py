# general modules
import json
import sys
import os
import numpy as np
from pathlib import Path

# learning framework
import torch
from torch.utils import data as torch_data
from torch.nn import functional as F
from torchvision import transforms

# config for experiments
from experiment_manager import args
from experiment_manager.config import config

# custom stuff
import evaluation_metrics as eval
import loss_functions as lf
import datasets
from utils import *

# networks from papers and ours
from networks.network_loader import load_network

# logging
import wandb


def setup(args):
    cfg = config.new_config()
    cfg.merge_from_file(f'configs/{args.config_file}.yaml')
    cfg.merge_from_list(args.opts)
    cfg.NAME = args.config_file
    return cfg


def train(net, cfg):
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    net.to(device)

    if cfg.TRAINER.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.0005)
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=cfg.TRAINER.LR, momentum=0.9)

    # loss functions
    if cfg.MODEL.LOSS_TYPE == 'BCEWithLogitsLoss':
        criterion = torch.nn.BCEWithLogitsLoss()
    elif cfg.MODEL.LOSS_TYPE == 'WeightedBCEWithLogitsLoss':
        positive_weight = torch.tensor([cfg.MODEL.POSITIVE_WEIGHT]).float().to(device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    elif cfg.MODEL.LOSS_TYPE == 'SoftDiceLoss':
        criterion = lf.soft_dice_loss
    elif cfg.MODEL.LOSS_TYPE == 'SoftDiceBalancedLoss':
        criterion = lf.soft_dice_loss_balanced
    elif cfg.MODEL.LOSS_TYPE == 'JaccardLikeLoss':
        criterion = lf.jaccard_like_loss
    elif cfg.MODEL.LOSS_TYPE == 'ComboLoss':
        criterion = lambda pred, gts: F.binary_cross_entropy_with_logits(pred, gts) + lf.soft_dice_loss(pred, gts)
    elif cfg.MODEL.LOSS_TYPE == 'WeightedComboLoss':
        criterion = lambda pred, gts: 2 * F.binary_cross_entropy_with_logits(pred, gts) + lf.soft_dice_loss(pred, gts)
    elif cfg.MODEL.LOSS_TYPE == 'FrankensteinLoss':
        criterion = lambda pred, gts: F.binary_cross_entropy_with_logits(pred, gts) + lf.jaccard_like_balanced_loss(
            pred, gts)
    elif cfg.MODEL.LOSS_TYPE == 'WeightedFrankensteinLoss':
        positive_weight = torch.tensor([cfg.MODEL.POSITIVE_WEIGHT]).float().to(device)
        criterion = lambda pred, gts: F.binary_cross_entropy_with_logits(pred, gts,
                                                                         pos_weight=positive_weight) + 5 * lf.jaccard_like_balanced_loss(
            pred, gts)
    else:
        criterion = lf.soft_dice_loss

    # reset the generators
    mode = cfg.DATASET.MODE
    dataset = datasets.OSCDDifferenceImages(cfg, 'train')
    drop_last = True
    batch_size = cfg.TRAINER.BATCH_SIZE
    dataloader_kwargs = {
        'batch_size': batch_size,
        'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
        'shuffle': cfg.DATALOADER.SHUFFLE,
        'drop_last': drop_last,
        'pin_memory': True,
    }
    if cfg.AUGMENTATION.OVERSAMPLING != 'none':
        dataloader_kwargs['sampler'] = dataset.sampler()
        dataloader_kwargs['shuffle'] = False

    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    save_path = Path(cfg.OUTPUT_BASE_DIR) / cfg.NAME
    save_path.mkdir(exist_ok=True)

    best_test_f1 = 0
    positive_pixels = 0
    pixels = 0
    global_step = 0
    epochs = cfg.TRAINER.EPOCHS
    batches = len(dataset) // batch_size if drop_last else len(dataset) // batch_size + 1
    for epoch in range(epochs):

        loss_tracker = 0
        net.train()

        for i, batch in enumerate(dataloader):

            diff_img = batch['diff_img'].to(device)
            label = batch['label'].to(device)

            optimizer.zero_grad()

            output = net(diff_img)

            loss = criterion(output, label)
            loss_tracker += loss.item()
            loss.backward()
            optimizer.step()

            positive_pixels += torch.sum(label).item()
            pixels += torch.numel(label)

            global_step += 1

        if epoch % cfg.LOGGING == 0:
            print(f'epoch {epoch} / {cfg.TRAINER.EPOCHS}')

            # printing and logging loss
            avg_loss = loss_tracker / batches
            print(f'avg training loss {avg_loss:.5f}')

            # positive pixel ratio used to check oversampling
            if cfg.DEBUG:
                print(f'positive pixel ratio: {positive_pixels / pixels:.3f}')
            else:
                wandb.log({f'positive pixel ratio': positive_pixels / pixels})
            positive_pixels = 0
            pixels = 0

            # model evaluation
            # train (different thresholds are tested)
            train_thresholds = torch.linspace(0, 1, 101).to(device)
            train_maxF1, train_maxTresh = model_evaluation(net, cfg, device, train_thresholds, run_type='train',
                                                           epoch=epoch, step=global_step)
            # test (using the best training threshold)
            test_threshold = torch.tensor([train_maxTresh])
            test_f1, _ = model_evaluation(net, cfg, device, test_threshold, run_type='test', epoch=epoch,
                                          step=global_step)

            if test_f1 > best_test_f1:
                print(f'BEST PERFORMANCE SO FAR! <--------------------', flush=True)
                best_test_f1 = test_f1

                if cfg.SAVE_MODEL and not cfg.DEBUG:
                    print(f'saving network', flush=True)
                    # model_file = save_path / 'best_net.pkl'
                    # torch.save(net.state_dict(), model_file)

            if (epoch + 1) == 390:
                if cfg.SAVE_MODEL and not cfg.DEBUG:
                    print(f'saving network', flush=True)
                    model_file = save_path / f'final_net.pkl'
                    torch.save(net.state_dict(), model_file)


def model_evaluation(net, cfg, device, thresholds, run_type, epoch, step):
    thresholds = thresholds.to(device)
    y_true_set = []
    y_pred_set = []

    measurer = eval.MultiThresholdMetric(thresholds)

    dataset = datasets.OSCDDifferenceImages(cfg, run_type, no_augmentation=True)
    dataloader_kwargs = {
        'batch_size': 1,
        'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
        'shuffle': cfg.DATALOADER.SHUFFLE,
        'pin_memory': True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    with torch.no_grad():
        net.eval()
        for step, batch in enumerate(dataloader):
            diff_img = batch['diff_img'].to(device)
            y_true = batch['label'].to(device)

            y_pred = net(diff_img)

            y_pred = torch.sigmoid(y_pred)

            y_true = y_true.detach()
            y_pred = y_pred.detach()
            y_true_set.append(y_true.cpu())
            y_pred_set.append(y_pred.cpu())

            measurer.add_sample(y_true, y_pred)

    print(f'Computing {run_type} F1 score ', end=' ', flush=True)

    f1 = measurer.compute_f1()
    fpr, fnr = measurer.compute_basic_metrics()
    maxF1 = f1.max()
    argmaxF1 = f1.argmax()
    best_fpr = fpr[argmaxF1]
    best_fnr = fnr[argmaxF1]
    best_thresh = thresholds[argmaxF1]

    if not cfg.DEBUG:
        wandb.log({
            f'{run_type} max F1': maxF1,
            f'{run_type} argmax F1': argmaxF1,
            f'{run_type} false positive rate': best_fpr,
            f'{run_type} false negative rate': best_fnr,
            'step': step,
            'epoch': epoch,
        })

    print(f'{maxF1.item():.3f}', flush=True)

    return maxF1.item(), best_thresh.item()


if __name__ == '__main__':

    # setting up config based on parsed argument
    parser = args.default_argument_parser()
    args = parser.parse_known_args()[0]
    cfg = setup(args)

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # loading network
    net = load_network(cfg)

    # tracking land with w&b
    if not cfg.DEBUG:
        wandb.init(
            name=cfg.NAME,
            project='urban_change_detection',
            tags=['run', 'change', 'detection', ],
        )

    try:
        train(net, cfg)
    except KeyboardInterrupt:
        print('Training terminated')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
