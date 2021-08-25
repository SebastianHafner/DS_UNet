
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchnet as tnt
import tools
from networks import network, networkL
import custom
from torch.utils.data import DataLoader
from pathlib import Path


if __name__ == '__main__':

    train_areas = ['abudhabi', 'beihai', 'aguasclaras', 'beirut', 'bercy', 'bordeaux', 'cupertino',
                   'hongkong', 'mumbai', 'nantes', 'rennes', 'saclay_e', 'pisa', 'rennes']

    # FOLDER = Path('C:/Users/hafne/urban_change_detection/data/Onera/')
    FOLDER = Path('/storage/shafner/urban_change_detection/Onera/')


    csv_file_train = FOLDER / 'myxys_train.csv'
    csv_file_val = FOLDER / 'myxys_val.csv'
    img_folder = FOLDER / 'images_preprocessed'  # folder with preprocessed images according to preprocess.py
    lbl_folder = FOLDER / 'labels'  # folder with OSCD dataset's labels
    save_folder = FOLDER / 'save_models'
    save_folder.mkdir(exist_ok=True)

    patch_size = 32

    # specify the number of dates you want to use, e.g put [1,2,3,4,5] if you want to use all five dates
    # or [1,2,5] to use just three of them
    nb_dates = [1, 2]

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)


    model_type = 'simple' #choose network type ('simple' or 'lstm')
                          #'simple' refers to a simple U-Net while 'lstm' refers to a U-Net involving LSTM blocks

    model_type = 'simple' #choose network type ('simple' or 'lstm')
                          #'simple' refers to a simple U-Net while 'lstm' refers to a U-Net involving LSTM blocks
    if model_type == 'simple':
        net = network.U_Net(4, 2, nb_dates)
    elif model_type == 'lstm':
        net = networkL.U_Net(4, 2, patch_size)
    else:
        net = None
        print('invalid on_network_argument')

    model = tools.to_cuda(net)

    change_dataset_train = custom.MyDataset(csv_file_train, train_areas, img_folder, lbl_folder, nb_dates, patch_size)
    change_dataset_val = custom.MyDataset(csv_file_val, train_areas, img_folder, lbl_folder, nb_dates, patch_size)
    mydataset_val = DataLoader(change_dataset_val, batch_size=32)

    # images_train, labels_train, images_val, labels_val = tools.make_data(size_len, portion, change_dataset)
    base_lr = 0.0001
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    weight_tensor = torch.FloatTensor(2)
    weight_tensor[0] = 0.20
    weight_tensor[1] = 0.80
    criterion = tools.to_cuda(nn.CrossEntropyLoss(tools.to_cuda(weight_tensor)))
    confusion_matrix = tnt.meter.ConfusionMeter(2, normalized=True)
    epochs = 60

    save_file = save_folder / 'progress_L2D.txt'
    save_file.touch(exist_ok=True)
    # ff = open(save_file, 'w')
    iter_ = 0
    for epoch in range(1, epochs + 1):
        mydataset = DataLoader(change_dataset_train, batch_size=32, shuffle=True)
        model.train()
        train_losses = []
        confusion_matrix.reset()

        for i, batch, in enumerate(mydataset):
            img_batch, lbl_batch = batch
            img_batch, lbl_batch = tools.to_cuda(img_batch.permute(1, 0, 2, 3, 4)), tools.to_cuda(lbl_batch)

            optimizer.zero_grad()
            output = model(img_batch.float())
            output_conf, target_conf = tools.conf_m(output, lbl_batch)
            confusion_matrix.add(output_conf, target_conf)

            loss = criterion(output, lbl_batch.long())
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

            del(img_batch, lbl_batch, loss)

        train_acc = (np.trace(confusion_matrix.conf) / float(np.ndarray.sum(confusion_matrix.conf))) * 100
        print(f'train loss: {np.mean(train_losses):.3f}, train acc: {train_acc:.3f}')
        confusion_matrix.reset()
        # end of epoch

        # VALIDATION
        with torch.no_grad():
            model.eval()

            val_losses = []
            print(len(mydataset_val))

            for i, batch, in enumerate(mydataset_val):
                # TODO: maybe fix this (last batch does not work)
                if i < (len(mydataset_val) - 1):
                    img_batch, lbl_batch = batch
                    img_batch, lbl_batch = tools.to_cuda(img_batch.permute(1, 0, 2, 3, 4)), tools.to_cuda(lbl_batch)

                    output = model(img_batch.float())
                    loss = criterion(output, lbl_batch.long())
                    val_losses.append(loss.item())
                    output_conf, target_conf = tools.conf_m(output, lbl_batch)
                    confusion_matrix.add(output_conf, target_conf)

            print(confusion_matrix.conf)
            test_acc = (np.trace(confusion_matrix.conf) / float(np.ndarray.sum(confusion_matrix.conf))) * 100
            change_acc = confusion_matrix.conf[1, 1] / float(confusion_matrix.conf[1, 0] + confusion_matrix.conf[1, 1]) * 100
            non_ch = confusion_matrix.conf[0, 0] / float(confusion_matrix.conf[0, 0]+confusion_matrix.conf[0, 1]) * 100
            print(f'val loss: {np.mean(val_losses):.3f}, val acc:  {test_acc:.3f}')
            print(f'Non_ch_Acc: {non_ch:.3f}, Change_Accuracy: {change_acc:.3f}')
            confusion_matrix.reset()

        # tools.write_results(ff, save_folder, epoch, train_acc, test_acc, change_acc, non_ch, np.mean(train_losses), np.mean(val_losses))
        if epoch % 5 == 0:  # save model every 5 epochs
            model_file = save_folder / f'model_{epoch}.pt'
            # torch.save(model.state_dict(), model_file)
