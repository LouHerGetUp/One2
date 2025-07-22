import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import copy
import time
from statistics import mean
from typing import List
import gc
from model import HGT
from utils import Metric_process, ROC, Metric


def train(device, model, data_loader, optimizer) -> float:
    model.train()
    for data in data_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        mask = data['flow'].train_mask
        loss = F.cross_entropy(out[mask], data['flow'].y[mask])
        loss.backward()
        optimizer.step()

    return float(loss)


def test(device, model, data_loader) -> List[float]:
    model.train()
    accs = []

    for split in ['train_mask', 'test_mask']:
        acc = []
        loss_list = []
        pred_total = torch.IntTensor().to(device)
        y_total = torch.IntTensor().to(device)

        for data in data_loader:
            data = data.to(device)

            test_start_time = time.time()
            mask = data['flow'][split]
            out = model(data)
            pred = out.argmax(dim=-1)
            test_end_time = time.time()
            test_time = test_end_time - test_start_time

            loss = F.cross_entropy(out[mask], data['flow'].y[mask]).item()

            acc.append(((pred[mask] == data['flow'].y[mask]).sum() / mask.sum()).item())
            loss_list.append(loss)

            pred_total = torch.cat([pred_total, pred[mask]], 0)
            y_total = torch.cat([y_total, data['flow'].y[mask]], 0)

        accs.append(float(mean(acc)))
        accs.append(y_total)
        accs.append(pred_total)
        accs.append(float(mean(loss_list)))
        if split == 'test_mask':
            accs.append(test_time)

    return accs


def test_roc(device, model, data_loader) -> List[float]:
    model.eval()
    accs = []

    with torch.no_grad():
        for split in ['test_mask']:
            y_total = torch.IntTensor().to(device)
            y_pred_total_roc = torch.FloatTensor().to(device)

            for data in data_loader:
                data = data.to(device)
                pred_roc = model(data)

                mask = data['flow'][split]

                y_pred_total_roc = torch.cat([y_pred_total_roc, pred_roc[mask]], 0)
                y_total = torch.cat([y_total, data['flow'].y[mask]], 0)

            accs.append(y_total)
            accs.append(y_pred_total_roc)

    return accs


def test_roc2(device, model, data_loader) -> List[float]:
    model.train()
    accs = []

    for split in ['train_mask', 'test_mask']:
        acc = []
        loss_list = []
        pred_total = torch.IntTensor().to(device)
        y_total = torch.IntTensor().to(device)

        for data in data_loader:
            data = data.to(device)

            test_start_time = time.time()
            mask = data['flow'][split]
            out = model(data)
            pred = out.argmax(dim=-1)
            test_end_time = time.time()
            test_time = test_end_time - test_start_time

            loss = F.cross_entropy(out[mask], data['flow'].y[mask]).item()

            acc.append(((pred[mask] == data['flow'].y[mask]).sum() / mask.sum()).item())
            loss_list.append(loss)

            pred_total = torch.cat([pred_total, pred[mask]], 0)
            y_total = torch.cat([y_total, data['flow'].y[mask]], 0)

        accs.append(float(mean(acc)))
        accs.append(y_total)
        accs.append(pred_total)
        accs.append(float(mean(loss_list)))
        if split == 'test_mask':
            accs.append(test_time)

    return accs


def fit(device, data_loader, learning_rate, hidden_channels, num_heads, num_layers, epochs, binary, SAVE_PATH,
        train_if, alg, dataset, file, roc):
    print("fit begin!")

    data_global = next(iter(data_loader))
    print("data_global done!")

    if binary == True:
        model = HGT(hidden_channels=hidden_channels, out_channels=2, num_heads=num_heads,
                    num_layers=num_layers, data_global=data_global)
    else:
        model = HGT(hidden_channels=hidden_channels, out_channels=15, num_heads=num_heads,
                    num_layers=num_layers, data_global=data_global)

    model = model.to(device)

    data_global = data_global.to(device)

    with torch.no_grad():
        out = model(data_global)

    print("Initialize lazy modules done!")

    if train_if == True:
        writer = SummaryWriter(log_dir="../logs/" + dataset + "_binary=" + str(binary) + "_loss_and_accuracy")

        print("writer done!")

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)

        print("optimizer done!")

        best_model_wts = copy.deepcopy(model.state_dict())
        best_val_acc = 0
        start_patience = patience = 10000000

        train_loss_list = []
        test_loss_list = []
        train_acc_list = []
        test_acc_list = []
        test_dr_list = []
        test_far_list = []
        test_f1_list = []
        train_time_list = []
        test_time_list = []

        print("train begin!")

        for epoch in range(1, epochs + 1):
            train_start_time = time.time()
            loss = train(device, model, data_loader, optimizer)
            train_end_time = time.time()
            train_time = train_end_time - train_start_time
            train_time = round(train_time, 7)

            train_acc, train_y_total, train_pred_total, train_loss, \
                test_acc, test_y_total, test_pred_total, test_loss, \
                test_time = test(device, model, data_loader)
            test_time = round(test_time * 1000, 2)

            epoch_DR, epoch_FAR, epoch_F1 = Metric_process(test_y_total, test_pred_total)

            train_loss_list.append(round(train_loss, 4))
            test_loss_list.append(round(test_loss, 4))
            train_acc_list.append(round(train_acc, 4))
            test_acc_list.append(round(test_acc, 4))
            test_dr_list.append(round(epoch_DR, 4))
            test_far_list.append(round(epoch_FAR, 4))
            test_f1_list.append(round(epoch_F1, 4))
            train_time_list.append(round(train_time, 2))
            test_time_list.append(test_time)

            writer.add_scalar("train_loss", loss, epoch)
            writer.add_scalar("train_accuracy", train_acc, epoch)
            writer.add_scalar("test_accuracy", test_acc, epoch)

            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
                  f'Test: {test_acc:.4f}, Train_time: {train_time:.2f}, Test_time: {test_time:.2f}')

            if test_acc > best_val_acc:
                patience = start_patience
                best_model_wts = copy.deepcopy(model.state_dict())
                best_val_acc = test_acc
            else:
                patience -= 1

            if patience <= 0:
                print('Stopping training as validation accuracy did not improve '
                      f'for {start_patience} epochs')
                break

            if epoch == epochs and roc == True:
                print('---------------------------------------')
                del train_y_total, train_pred_total, test_y_total, test_pred_total
                gc.collect()
                torch.cuda.empty_cache()
                print('---------------------------------------')
                test_y_total, test_y_pred_total_roc = test_roc(device, model, data_loader)
                ROC(test_y_total.detach(), test_y_pred_total_roc.detach(), alg, dataset, file)
                print('---------------------------------------')

        print("train done!")

        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(), SAVE_PATH)

        process_list = [train_loss_list, test_loss_list, train_acc_list, test_acc_list,
                        test_dr_list, test_far_list, test_f1_list, train_time_list, test_time_list]
        process_df = pd.DataFrame(process_list,
                                  index=['train_loss_list', 'test_loss_list', 'train_acc_list',
                                         'test_acc_list', 'test_dr_list', 'test_far_list',
                                         'test_f1_list', 'train_time_list', 'test_time_list'])
        process_df.to_csv(
            f"output/process_df+dataset={dataset}+file={file}+learning_rate={learning_rate}+hidden_channels={hidden_channels}+num_heads={num_heads}+num_layers={num_layers}.csv")
        print(process_list)
        print(process_df)
        writer.close()

    else:
        model.load_state_dict(torch.load(SAVE_PATH))

        if roc == True:
            test_y_total, test_y_pred_total_roc = test_roc(device, model, data_loader)
        else:
            train_acc, train_y_total, train_pred_total, test_acc, test_y_total, test_pred_total, test_time = test(
                device, model, data_loader)
            print("--------------------------------------train----------------------------------------------")
            Metric(train_y_total, train_pred_total)

        print("--------------------------------------test----------------------------------------------")
        Metric(test_y_total, test_pred_total)

        if roc == True:
            ROC(test_y_total.detach(), test_y_pred_total_roc.detach(), alg, dataset, file)
        else:
            print(f'Train: {train_acc:.4f}, Test: {test_acc:.4f}')