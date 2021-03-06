# general modules
import sys
import os
import numpy as np
from pathlib import Path

# learning framework
import torch
from torch.utils import data as torch_data

from utils import experiment_manager, networks, loss_functions, datasets, evaluation_metrics, paths

# logging
import wandb


def train(net, cfg):

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.0005)

    criterion = loss_functions.get_loss_function(cfg, device)

    # reset the generators
    dataset = datasets.OSCDDataset(cfg, 'train')
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

    # setting up save dir
    dirs = paths.load_paths()
    net_dir = Path(dirs.OUTPUT_ROOT) / 'run_logs'
    net_dir.mkdir(exist_ok=True)

    positive_pixels = 0
    pixels = 0
    global_step = 0
    epochs = cfg.TRAINER.EPOCHS
    batches = len(dataset) // batch_size if drop_last else len(dataset) // batch_size + 1
    for epoch in range(epochs):

        loss_tracker = 0
        net.train()

        for i, batch in enumerate(dataloader):

            t1_img = batch['t1_img'].to(device)
            t2_img = batch['t2_img'].to(device)

            label = batch['label'].to(device)

            optimizer.zero_grad()

            output = net(t1_img, t2_img)

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

            if (epoch + 1) == epochs:
                if cfg.SAVE_MODEL and not cfg.DEBUG:
                    print(f'saving network', flush=True)
                    net_file = net_dir / cfg.NAME / f'final_net.pkl'
                    net_file.parent.mkdir(exist_ok=True)
                    torch.save(net.state_dict(), net_file)


def model_evaluation(net, cfg, device, thresholds, run_type, epoch, step):
    thresholds = thresholds.to(device)
    y_true_set = []
    y_pred_set = []

    measurer = evaluation_metrics.MultiThresholdMetric(thresholds)

    dataset = datasets.OSCDDataset(cfg, run_type, no_augmentation=True)
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
            t1_img = batch['t1_img'].to(device)
            t2_img = batch['t2_img'].to(device)
            y_true = batch['label'].to(device)

            y_pred = net(t1_img, t2_img)

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
    parser = experiment_manager.default_argument_parser()
    args = parser.parse_known_args()[0]
    cfg = experiment_manager.setup(args)

    # deterministic training
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # loading network
    net = networks.create_network(cfg)

    # tracking land with w&b
    if not cfg.DEBUG:
        wandb.init(
            name=cfg.NAME,
            project='urban_change_detection',
            tags=['run', 'change', 'detection', ],
        )

    # here we go
    try:
        train(net, cfg)
    except KeyboardInterrupt:
        print('Training terminated')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


