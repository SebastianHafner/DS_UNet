import torch
from torch.utils import data as torch_data

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from utils import networks, datasets, experiment_manager, evaluation_metrics, paths


def visual_evaluation(cfg_name: str, dataset: str = 'test', label_pred_only: bool = False):

    dirs = paths.load_paths()

    cfg_file = Path(dirs.HOME_ROOT) / 'configs' / f'{cfg_name}.yaml'
    cfg = experiment_manager.load_cfg(cfg_file)

    net_file = Path(dirs.OUTPUT_ROOT) / 'run_logs' / cfg_name / f'final_net.pkl'
    net = networks.load_network(cfg, net_file)

    mode = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(mode)

    # bands for visualizaiton
    s1_bands, s2_bands = cfg.DATASET.SENTINEL1_BANDS, cfg.DATASET.SENTINEL2_BANDS
    all_bands = s1_bands + s2_bands

    dataset = datasets.OSCDDataset(cfg, dataset, no_augmentation=True)
    dataloader_kwargs = {
        'batch_size': 1,
        'num_workers': 0,
        'shuffle': False,
        'pin_memory': True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    with torch.no_grad():
        net.eval()
        for step, batch in enumerate(dataloader):
            city = batch['city'][0]
            print(city)
            t1_img = batch['t1_img'].to(device)
            t2_img = batch['t2_img'].to(device)
            y_true = batch['label'].to(device)
            y_pred = net(t1_img, t2_img)
            y_pred = torch.sigmoid(y_pred)
            y_pred = y_pred.cpu().detach().numpy()[0, ]
            y_pred = y_pred > cfg.THRESH
            y_pred = y_pred.transpose((1, 2, 0)).astype('uint8')

            # label
            y_true = y_true.cpu().detach().numpy()[0, ]
            y_true = y_true.transpose((1, 2, 0)).astype('uint8')

            if label_pred_only:
                fig, axs = plt.subplots(1, 2, figsize=(10, 10))
                axs[0].imshow(y_true[:, :, 0])
                axs[1].imshow(y_pred[:, :, 0])
            else:
                fig, axs = plt.subplots(1, 4, figsize=(20, 10))
                rgb_indices = [all_bands.index(band) for band in ('B04', 'B03', 'B02')]
                for i, img in enumerate([t1_img, t2_img]):
                    img = img.cpu().detach().numpy()[0, ]
                    img = img.transpose((1, 2, 0))
                    rgb = img[:, :, rgb_indices] / 0.3
                    rgb = np.minimum(rgb, 1)
                    axs[i+2].imshow(rgb)
                axs[0].imshow(y_true[:, :, 0])
                axs[1].imshow(y_pred[:, :, 0])

            for ax in axs:
                ax.set_axis_off()

            evaluation_dir = Path(dirs.OUTPUT_ROOT) / 'evaluation'
            evaluation_dir.mkdir(exist_ok=True)

            save_dir = evaluation_dir / cfg_name
            save_dir.mkdir(exist_ok=True)
            file = save_dir / f'eval_{cfg_name}_{city}.png'

            plt.savefig(file, dpi=300, bbox_inches='tight')
            plt.close()


def numeric_evaluation(cfg_name: str):

    dirs = paths.load_paths()

    cfg_file = Path(dirs.HOME_ROOT) / 'configs' / f'{cfg_name}.yaml'
    cfg = experiment_manager.load_cfg(cfg_file)

    net_file = Path(dirs.OUTPUT_ROOT) / 'run_logs' / cfg_name / f'final_net.pkl'
    net = networks.load_network(cfg, net_file)

    mode = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(mode)

    dataset = datasets.OSCDDataset(cfg, 'test', no_augmentation=True)
    dataloader_kwargs = {
        'batch_size': 1,
        'num_workers': 0,
        'shuffle': cfg.DATALOADER.SHUFFLE,
        'pin_memory': True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    tta_thresholds = np.linspace(0, 1, 11)

    def predict(t1, t2):
        pred = net(t1, t2)
        pred = torch.sigmoid(pred) > cfg.THRESH
        pred = pred.detach().float()
        return pred

    def evaluate(true, pred):
        f1_score = evaluation_metrics.f1_score(true.flatten(), pred.flatten(), dim=0).item()
        true_pos = evaluation_metrics.true_pos(true.flatten(), pred.flatten(), dim=0).item()
        false_pos = evaluation_metrics.false_pos(true.flatten(), pred.flatten(), dim=0).item()
        false_neg = evaluation_metrics.false_neg(true.flatten(), pred.flatten(), dim=0).item()
        return f1_score, true_pos, false_pos, false_neg

    cities, f1_scores, true_positives, false_positives, false_negatives = [], [], [], [], []
    tta = []
    with torch.no_grad():
        net.eval()
        for step, batch in enumerate(dataloader):

            city = batch['city'][0]
            print(city)
            cities.append(city)

            t1_img = batch['t1_img'].to(device)
            t2_img = batch['t2_img'].to(device)

            y_true = batch['label'].to(device)

            y_pred = predict(t1_img, t2_img)
            f1_score, tp, fp, fn = evaluate(y_true, y_pred)
            f1_scores.append(f1_score)
            true_positives.append(tp)
            false_positives.append(fp)
            false_negatives.append(fn)

            sum_preds = torch.zeros(y_true.shape).float().to(device)
            n_augs = 0

            # rotations
            for k in range(4):
                t1_img_rot = torch.rot90(t1_img, k, (2, 3))
                t2_img_rot = torch.rot90(t2_img, k, (2, 3))
                y_pred = predict(t1_img_rot, t2_img_rot)
                y_pred = torch.rot90(y_pred, 4 - k, (2, 3))

                sum_preds += y_pred
                n_augs += 1

            # flips
            for flip in [(2, 3), (3, 2)]:
                t1_img_flip = torch.flip(t1_img, flip)
                t2_img_flip = torch.flip(t1_img, flip)
                y_pred = predict(t1_img_flip, t2_img_flip)
                y_pred = torch.flip(y_pred, flip)

                sum_preds += y_pred
                n_augs += 1

            pred_tta = sum_preds.float() / n_augs
            tta_city = []
            for ts in tta_thresholds:
                y_pred = pred_tta > ts
                y_pred = y_pred.float()
                eval_ts = evaluate(y_true, y_pred)
                tta_city.append(eval_ts)
            tta.append(tta_city)

        precision = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_positives))
        recall = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_negatives))
        f1_score = 2 * (precision * recall / (precision + recall))
        print(f'precision: {precision:.3f}, recall: {recall:.3f}, f1: {f1_score:.3f}')

        tta_f1_scores = []
        for i, ts in enumerate(tta_thresholds):
            tta_ts = [city[i] for city in tta]
            tp = np.sum([eval_ts[1] for eval_ts in tta_ts])
            fp = np.sum([eval_ts[2] for eval_ts in tta_ts])
            fn = np.sum([eval_ts[3] for eval_ts in tta_ts])
            pre_tta = tp / (tp + fp + 1e-5)
            re_tta = tp / (tp + fn + 1e-5)
            f1_score_tta = 2 * (pre_tta * re_tta / (pre_tta + re_tta + 1e-5))
            tta_f1_scores.append(f1_score_tta)
            print(f'{ts:.2f}: {f1_score_tta:.3f}')

        fig, ax = plt.subplots()
        ax.plot(tta_thresholds, tta_f1_scores)
        ax.plot(tta_thresholds, [f1_score] * 11, label=f'without tta ({f1_score:.3f})')
        ax.legend()
        ax.set_xlabel('tta threshold (gt)')
        ax.set_ylabel('f1 score')
        ax.set_title(cfg_file.stem)
        # plt.show()


def visualize_missclassifications(cfg_name: str):

    dirs = paths.load_paths()

    cfg_file = Path(dirs.HOME_ROOT) / 'configs' / f'{cfg_name}.yaml'
    cfg = experiment_manager.load_cfg(cfg_file)

    net_file = Path(dirs.OUTPUT_ROOT) / 'run_logs' / cfg_name / f'final_net.pkl'
    net = networks.load_network(cfg, net_file)

    mode = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(mode)

    dataset = datasets.OSCDDataset(cfg, 'test', no_augmentation=True)
    dataloader_kwargs = {
        'batch_size': 1,
        'num_workers': 0,
        'shuffle': False,
        'pin_memory': True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    with torch.no_grad():
        net.eval()
        for step, batch in enumerate(dataloader):
            city = batch['city'][0]
            print(city)
            t1_img = batch['t1_img'].to(device)
            t2_img = batch['t2_img'].to(device)
            y_true = batch['label'].to(device)
            y_pred = net(t1_img, t2_img)
            y_pred = torch.sigmoid(y_pred)
            y_pred = y_pred.cpu().detach().numpy()[0, ]
            y_pred = y_pred > cfg.THRESH
            y_pred = y_pred.transpose((1, 2, 0)).astype('uint8')[:, :, 0]

            # label
            y_true = y_true.cpu().detach().numpy()[0, ]
            y_true = y_true.transpose((1, 2, 0)).astype('uint8')[:, :, 0]

            img = np.zeros((*y_true.shape, 3))
            true_positives = np.logical_and(y_pred, y_true)
            false_positives = np.logical_and(y_pred, np.logical_not(y_true))
            false_negatives = np.logical_and(np.logical_not(y_pred), y_true)
            img[true_positives, :] = [1, 1, 1]
            img[false_positives] = [0, 1, 0]
            img[false_negatives] = [1, 0, 1]

            fig, ax = plt.subplots()
            ax.imshow(img)
            ax.set_axis_off()

            evaluation_dir = Path(dirs.OUTPUT_ROOT) / 'evaluation'
            evaluation_dir.mkdir(exist_ok=True)

            save_dir = evaluation_dir / cfg_name
            save_dir.mkdir(exist_ok=True)
            file = save_dir / f'missclassfications_{cfg_name}_{city}.png'

            plt.savefig(file, dpi=300, bbox_inches='tight')
            plt.close()


def standard_deviation(cfg_name: str):

    dirs = paths.load_paths()

    cfg_file = Path(dirs.HOME_ROOT) / 'configs' / f'{cfg_name}.yaml'
    cfg = experiment_manager.load_cfg(cfg_file)

    net_file = Path(dirs.OUTPUT_ROOT) / 'run_logs' / cfg_name / f'final_net.pkl'
    net = networks.load_network(cfg, net_file)

    mode = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(mode)

    dataset = datasets.OSCDDataset(cfg, 'test', no_augmentation=True)
    f1_scores = {}
    with torch.no_grad():
        for index in range(len(dataset)):
            item = dataset.__getitem__(index)
            t1_img, t2_img = item['t1_img'].to(device), item['t2_img'].to(device)
            logits = net(t1_img.unsqueeze(0), t2_img.unsqueeze(0))
            y_pred = torch.sigmoid(logits) > cfg.THRESH
            y_pred = y_pred.flatten().float()
            y_true = item['label'].to(device).flatten().float()

            tp = torch.sum(torch.logical_and(y_pred, y_true)).item()
            fp = torch.sum(torch.logical_and(y_pred, torch.logical_not(y_true))).item()
            fn = torch.sum(torch.logical_and(torch.logical_not(y_pred), y_true)).item()

            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * (precision * recall) / (precision + recall)

            city = item['city']
            f1_scores[city] = f1

    print(f1_scores)
    print(np.std(list(f1_scores.values())))


if __name__ == '__main__':

    cfg_name = 'fusion_dualstream_7'

    visual_evaluation(cfg_name, 'test', label_pred_only=False)
    numeric_evaluation(cfg_name)
    visualize_missclassifications(cfg_name)
    standard_deviation(cfg_name)
