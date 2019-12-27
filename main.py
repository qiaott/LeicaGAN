from cfg.config import cfg, cfg_from_file
from datasets import TextDataset
from trainer import condGANTrainer as trainer

import os
import sys
import time
import random
import pprint
import argparse
import numpy as np
from nltk.tokenize import RegexpTokenizer
import torch
import torchvision.transforms as transforms

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a AttnGAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/Ad_Class.yml', type=str)
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args

def gen_example(wordtoix, algo):
    '''generate images from example sentences'''
    filepath = 'example_filenames.txt'
    data_dic = {}
    with open(filepath, "r") as f:
        filenames = f.read().split('\n')
        for name in filenames:
            if len(name) == 0:
                continue
            filepath = '%s.txt' % (name)
            with open(filepath, "r") as f_:
                print('Load from:', name)
                sentences = f_.read().split('\n')
                # a list of indices for a sentence
                captions = []
                cap_lens = []
                for sent in sentences:
                    print('sent:', sent)
                    if len(sent) == 0:
                        continue
                    sent = sent.replace("\\ufffd\\ufffd", " ")
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(sent.lower())
                    if len(tokens) == 0:
                        print('sent', sent)
                        continue

                    rev = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0 and t in wordtoix:
                            rev.append(wordtoix[t])
                    captions.append(rev)
                    cap_lens.append(len(rev))
            max_len = np.max(cap_lens)

            sorted_indices = np.argsort(cap_lens)[::-1]
            cap_lens = np.asarray(cap_lens)
            cap_lens = cap_lens[sorted_indices]
            cap_array = np.zeros((len(captions), max_len), dtype='int64')
            for i in range(len(captions)):
                idx = sorted_indices[i]
                cap = captions[idx]
                c_len = len(cap)
                cap_array[i, :c_len] = cap
            key = name[(name.rfind('/') + 1):]
            data_dic[key] = [cap_array, cap_lens, sorted_indices]
    algo.gen_example(data_dic)

if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    assert torch.cuda.is_available(), print('Ops, with no GPUs? It is sad..')

    print('Using config:')
    pprint.pprint(cfg)

    # if not cfg.TRAIN.FLAG:
    #     args.manualSeed = 100
    # elif args.manualSeed is None:
    #     args.manualSeed = random.randint(1, 10000)
    # random.seed(args.manualSeed)
    # np.random.seed(args.manualSeed)
    # torch.manual_seed(args.manualSeed)
    # if cfg.CUDA:
    #     torch.cuda.manual_seed_all(args.manualSeed)

    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed(args.manualSeed)
        torch.cuda.manual_seed_all(args.manualSeed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def worker_init_fn(worker_id):  # After creating the workers, each worker has an independent seed that is initialized to the curent random seed + the id of the worker
        np.random.seed(args.manualSeed + worker_id)

    output_dir = '%s/%s_%s/%s' % \
        (cfg.OUTPUT_DIR, cfg.DATASET_NAME, cfg.GAN.GNET, cfg.CONFIG_NAME)

    split_dir, bshuffle = 'train', True
    if not cfg.TRAIN.FLAG:
        split_dir = 'test'

    # Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    image_transform = transforms.Compose([
        transforms.Resize((int(imsize * 72 / 64), int(imsize * 72 / 64))),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])

    dataset = TextDataset(os.path.join(cfg.DATA_DIR, cfg.DATASET_NAME), split_dir,
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))

    # Define models and go to train/evaluate
    algo = trainer(output_dir, dataloader, dataset.n_words, dataset.ixtoword)

    start_t = time.time()
    if cfg.TRAIN.FLAG:
        algo.train()
    elif cfg.B_VALIDATION:
        algo.sampling(split_dir)  # generate images for the whole test dataset
    elif cfg.MODE == 'PersonalizedGeneration':
        gen_example(dataset.wordtoix, algo)  # generate images for customized captions
    else:
        print('Please specify the mode of the experiment...')
    end_t = time.time()
    print('Total time for training:', end_t - start_t)
