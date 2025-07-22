import argparse
import os
import torch
from data_process import data_process
from trainer import fit
from utils import setup_seed


def process_and_fit(args):
    alg = args.alg
    dataset = args.dataset
    binary = args.binary
    batchsize = args.batchsize
    epochs = args.epochs
    cuda = args.cuda
    train_if = args.train_if
    roc = args.roc
    device = torch.device(cuda)
    learning_rate = args.learning_rate
    hidden_channels = args.hidden_channels
    num_heads = args.num_heads
    num_layers = args.num_layers
    file = args.file

    path = "../dataset/" + dataset + "/" + file
    SAVE_PATH = "./model/" + __file__.split("/")[-1].split(".")[0] + ".pth"
    data_loader = data_process(path=path, dataset=dataset, binary=binary, batchsize=batchsize)

    fit(device, data_loader, learning_rate=learning_rate, hidden_channels=hidden_channels,
        num_heads=num_heads, num_layers=num_layers, epochs=epochs, binary=binary, SAVE_PATH=SAVE_PATH, train_if=train_if, alg=alg, dataset=dataset,
        file=file, roc=roc)


if __name__ == "__main__":
    ALG = ['HGT']
    DATA = ['CIC-IDS2017', 'CAR-HACKING', 'TON_IoT', 'CAN-intrusion', 'CIC-UNSW-NB15', 'CICIoV2024']

    p = argparse.ArgumentParser()
    p.add_argument('--alg',
                   help='algorithm to use.',
                   default='HGT',
                   choices=ALG)
    p.add_argument('--dataset',
                   help='Experimental dataset.',
                   type=str,
                   default='CIC-IDS2017',
                   choices=DATA)
    p.add_argument('--file',
                   help='Experimental file.',
                   type=str,
                   default='df.csv')
    p.add_argument('--binary',
                   help='Perform binary or muticlass task',
                   type=bool,
                   default=True)
    p.add_argument('--batchsize',
                   help='batchsize for dataloader.',
                   type=int,
                   default=2 ** 7)
    p.add_argument('--epochs',
                   help='epochs for train.',
                   type=int,
                   default=1)
    p.add_argument('--cuda',
                   help='cuda device.',
                   type=str,
                   default='cuda:0')
    p.add_argument('--train_if',
                   help='train or not.',
                   type=bool,
                   default=True)
    p.add_argument('--roc',
                   help='roc or not.',
                   type=bool,
                   default=True)
    p.add_argument('--learning_rate',
                   help='learning_rate.',
                   type=float,
                   default=0.01)
    p.add_argument('--hidden_channels',
                   help='hidden_channels.',
                   type=int,
                   default=64)
    p.add_argument('--num_heads',
                   help='num_heads.',
                   type=int,
                   default=2)
    p.add_argument('--num_layers',
                   help='num_layers.',
                   type=int,
                   default=1)

    args = p.parse_args()

    setup_seed(1999)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    device_ids = [0, 1, 2, 3]
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    process_and_fit(args)