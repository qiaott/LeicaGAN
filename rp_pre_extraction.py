from miscc.utils import mkdir_p, build_super_images, parse_str, save_pickle
from cfg.config import cfg, cfg_from_file
from datasets import prepare_data
from model import RNN_ENCODER, CNN_ENCODER
from os import path
import sys
import pprint
import argparse
import os
from cfg.config import cfg
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import numpy.random as random
import pickle
from PIL import Image
from torch.autograd import Variable
from miscc.utils import load_pickle

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

UPDATE_INTERVAL = 20

def load_imgs(img_path):
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    normalize = transforms.Compose([
        transforms.Resize((int(imsize * 72 / 64), int(imsize * 72 / 64))),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img = Image.open(img_path).convert('RGB')
    norm_img = normalize(img)
    return norm_img


def prepare_data(data):
    # index, captions, cap_lens = data
    index, captions, cap_lens, cls_id, imgs = data
    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = torch.sort(cap_lens, 0, True)
    captions = captions[sorted_cap_indices].squeeze()

    imgs = imgs[sorted_cap_indices]

    if cfg.CUDA:
        captions = Variable(captions).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
        imgs.cuda()
    else:
        captions = Variable(captions)
        sorted_cap_lens = Variable(sorted_cap_lens)

    return [index, captions, sorted_cap_lens, imgs, cls_id]


class TextDataset(data.Dataset):
    def __init__(self, data_dir, base_size=64,
                 transform=None, target_transform=None):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE
        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2
        self.data = []
        self.data_dir = data_dir
        self.ixtoword, self.wordtoix, self.n_words = self.load_dictionary(data_dir+'/dictionary.pickle')
        dataset_path = self.data_dir + '/test_dataset.pickle'
        self.cls2imgid = load_pickle(data_dir + '/test_cls2imgid.pickle')
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
            self.dataset = dataset
            print('len of datatset:', len(self.dataset))
            del dataset
            print('Load dataset from:', dataset_path)

    def load_dictionary(self, dict_path):
        with open(dict_path, 'rb') as f:
            ixtoword, wordtoix, n_words = pickle.load(f, encoding='iso-8859-1')
            return [ixtoword, wordtoix, n_words]

    def get_caption(self, caption):
        # a list of indices for a sentence
        sent_caption = np.asarray(caption).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = cfg.TEXT.WORDS_NUM
        return x, x_len

    def get_img_info(self, img_index):
        img_data = self.dataset[img_index]
        key = img_data['img_name']
        cls_id = int(str(img_data['cls_index']))
        bbox = None

        if cfg.DATASET_NAME == 'flower':
            img_name = '%s/images/%s.jpg' % (self.data_dir, key)
            key = key + '.jpg'
        else:
            img_name = '%s/images/%s/%s' % (self.data_dir, img_data['img_class'], key)
        imgs = load_imgs(img_name)
        return key, cls_id, bbox, img_name, imgs

    def __getitem__(self, index): # index of text
        img_index = index // cfg.TEXT.CAPTIONS_PER_IMAGE
        text_index = index % cfg.TEXT.CAPTIONS_PER_IMAGE
        key, cls_id, bbox, name, imgs = self.get_img_info(img_index)
        current_captions = self.dataset[img_index]['text'][text_index]
        current_caps, current_cap_len = self.get_caption(current_captions)

        return img_index, current_caps, current_cap_len, cls_id, imgs

    def __len__(self):
        return len(self.dataset)*cfg.TEXT.CAPTIONS_PER_IMAGE


def parse_args():
    parser = argparse.ArgumentParser(description='extract features of image for calculating the R precision score')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='..', type=str)
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


def evaluate(dataloader, cnn_model, rnn_model, batch_size):
    cnn_model.eval()
    rnn_model.eval()
    count = 0
    text_features = {}
    for step, data in enumerate(dataloader, 0):
        index, captions, cap_lens, _, _ = prepare_data(data)
        batch_size = captions.size()[0]
        hidden = rnn_model.init_hidden(batch_size)
        _, sent_emb = rnn_model(captions, cap_lens, hidden)
        index.data.cpu().numpy()
        sent_emb.data.cpu().numpy()
        for i in range(batch_size):
            idx = index[i].item()
            if idx not in text_features.keys():
                text_features[idx] = [sent_emb[i].data.cpu().numpy()]
            else:
                text_features[idx].append(sent_emb[i].data.cpu().numpy())
            count += 1
    print('processed %d data'%count) # 29322
    return text_features


def img2info(dataloader, cnn_model, rnn_model, batch_size):
    cnn_model.eval()
    rnn_model.eval()
    count = 0
    text_features = {}
    for step, data in enumerate(dataloader, 0):
        index, captions, cap_lens, imgs, cls_id = prepare_data(data)
        hidden = rnn_model.init_hidden(batch_size)
        _, sent_emb = rnn_model(captions, cap_lens, hidden)
        _, img_emb = cnn_model(imgs)
        index.data.cpu().numpy()
        cls_id.data.cpu().numpy()
        sent_emb.data.cpu().numpy()
        img_emb.data.cpu().numpy()
        for i in range(batch_size):
            idx = index[i].item()
            if idx not in text_features.keys():
                # print(idx)
                text_features[idx] = [sent_emb[i].data.cpu().numpy()]
            else:
                # print('same')
                text_features[idx].append(sent_emb[i].data.cpu().numpy())
            count += 1
    print('processed %d data'%count)
    return text_features


def build_models():
    text_encoder = RNN_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
    if cfg.TRAIN.NET_E != '':
        state_dict = torch.load(cfg.TRAIN.NET_E)
        text_encoder.load_state_dict(state_dict)
        print('Load', cfg.TRAIN.NET_E)
        name = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = torch.load(name)
        image_encoder.load_state_dict(state_dict)
        print('Load', name)
        istart = cfg.TRAIN.NET_E.rfind('_') + 8
        iend = cfg.TRAIN.NET_E.rfind('.')
        start_epoch = cfg.TRAIN.NET_E[istart:iend]
        start_epoch = int(start_epoch) + 1
        print('start_epoch', start_epoch)
    if cfg.CUDA:
        assert (torch.cuda.is_available())
        text_encoder.cuda()
        image_encoder.cuda()
    return text_encoder, image_encoder


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    cfg.GPU_ID = parse_str(cfg.GPU_ID)
    torch.cuda.set_device(cfg.GPU_ID[0])
    output_dir = '%s/%s_%s' % (cfg.OUTPUT_DIR, cfg.DATASET_NAME, cfg.ENCODER1)
    model_dir = os.path.join(output_dir, 'Model')
    # image_dir = os.path.join(output_dir, 'Image')
    mkdir_p(model_dir)
    # mkdir_p(image_dir)
    expect_image_encoder = os.path.join(model_dir, 'image_encoder200.pth')
    expect_text_encoder = os.path.join(model_dir, 'text_encoder200.pth')
    print(expect_image_encoder)
    print(expect_text_encoder)
    assert path.exists(expect_image_encoder)
    assert path.exists(expect_text_encoder)

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
    batch_size = cfg.TRAIN.BATCH_SIZE
    dataset = TextDataset(os.path.join(cfg.DATA_DIR, cfg.DATASET_NAME),
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=None)

    print('dataset.n_words, dataset.embeddings_num:', dataset.n_words, dataset.embeddings_num)
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, drop_last=False,
        shuffle=False, num_workers=int(cfg.WORKERS))
    text_encoder, image_encoder = build_models()
    text_features = evaluate(dataloader, image_encoder, text_encoder, batch_size)
    print(len(text_features.keys())) # 29330
    assert len(text_features.keys()) == len(dataset) / 10
    save_path_textfeature = os.path.join(model_dir, 'testimg2textfeature.pickle')
    save_pickle(text_features, save_path_textfeature)
    print('Congrats, save pickle to %s!'%(save_path_textfeature))