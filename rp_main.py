from cfg.config import cfg, cfg_from_file
from rp_dataset import RPDataset
from rp_trainer import condGANTrainer as Trainer
import os
import sys
import time
import random
import pprint
import argparse
import numpy as np
import torch

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), '././.')))
sys.path.append(dir_path)


def parse_args():
    parser = argparse.ArgumentParser(description='this is for calculating the R precision')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='..', type=str)
    parser.add_argument('--w', type=int, default=100, help='manual seed')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    assert torch.cuda.is_available(), 'No GPUs..'

    print('Using config:')
    pprint.pprint(cfg)
    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    split_dir, bshuffle = 'train', True
    if not cfg.TRAIN.FLAG:
        split_dir = 'test'
        bshuffle = False

    output_dir = '%s/%s_%s/%s' % (cfg.OUTPUT_DIR, cfg.DATASET_NAME, cfg.GAN.GNET, cfg.CONFIG_NAME)
    encoder_dir = '%s/%s_%s/Model' % (cfg.OUTPUT_DIR, cfg.DATASET_NAME, cfg.ENCODER1)
    img2textfeature_file = os.path.join(encoder_dir, 'testimg2textfeature.pickle')
    print(img2textfeature_file)
    assert os.path.isfile(img2textfeature_file), 'run rp_pre_extraction first!'
    # Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    dataset = RPDataset(os.path.join(cfg.DATA_DIR, cfg.DATASET_NAME), base_size=cfg.TREE.BASE_SIZE, transform=None)
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS), worker_init_fn=worker_init_fn)

    algo = Trainer(output_dir, dataloader, dataset.n_words, dataset.ixtoword)

    start_t = time.time()
    algo.R_precision(split_dir)
    end_t = time.time()
    print('Total time for training:', end_t - start_t)
