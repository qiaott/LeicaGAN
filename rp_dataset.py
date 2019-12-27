#coding=utf-8
import os
from cfg.config import cfg
import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import pickle
from miscc.utils import load_pickle

def prepare_RP_data(data):
    caps, cap_len, A_keys, A_cls_id = data

    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(cap_len, 0, True)

    captions = caps[sorted_cap_indices].squeeze()
    # A_cls_id = A_cls_id[sorted_cap_indices].numpy()
    A_keys = [A_keys[i] for i in sorted_cap_indices.numpy()]
    if cfg.CUDA:
        captions = Variable(captions).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
    else:
        captions = Variable(captions)
        sorted_cap_lens = Variable(sorted_cap_lens)

    return [captions, sorted_cap_lens, A_keys, A_cls_id]

def get_mis_99(current_cls_id, cls2imgid, img2textfeature):
    # print('current_cls_id:', current_cls_id)
    all_cls = list(cls2imgid.keys())
    # print('all_cls:', all_cls)
    cls_p = (np.array(all_cls) != current_cls_id) / (len(all_cls) - 1)
    # print('cls_p:', cls_p)
    diff_cls_idx = np.random.choice(all_cls, 99, p=cls_p)
    # print('diff_cls_idx:', diff_cls_idx)
    texts = []
    for idx in diff_cls_idx:
        # print('idx:', idx)
        img_idx = np.random.choice(cls2imgid[idx], 1)
        img_idx = img_idx[0]
        print('img_idx:', img_idx)
        random_text_id = np.random.randint(0, 10)
        print('random_text_id:', random_text_id)
        text_feature = img2textfeature[img_idx][random_text_id]
        texts.append(text_feature)
    assert len(texts) == 99
    return texts


def load_imgs(img_path):
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img = Image.open(img_path).convert('RGB')
    norm_img = normalize(img)
    return norm_img

class RPDataset(data.Dataset):
    def __init__(self, data_dir, base_size=64, transform=None):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE-1
        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2
        self.data = []
        self.data_dir = data_dir
        self.ixtoword, self.wordtoix, self.n_words = self.load_dictionary(data_dir+'/dictionary.pickle')
        dataset_path = self.data_dir + '/test_dataset.pickle'
        self.cls2imgid = load_pickle(data_dir + '/test_cls2imgid.pickle')
        encoder_dir = '%s/%s_%s/Model' % (cfg.OUTPUT_DIR, cfg.DATASET_NAME, cfg.ENCODER1)
        self.img2textfeature = load_pickle(os.path.join(encoder_dir, 'testimg2textfeature.pickle'))
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

    # load_imgs
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

    def __getitem__(self, A_index):
        # get A1: data, key, cls_id, bbox
        A_key, A_cls_id, A_bbox, A_name, A_imgs = self.get_img_info(A_index)
        # random select a sentence for A1
        random_ix = np.random.randint(0, self.embeddings_num)
        current_captions = self.dataset[A_index]['text'][random_ix]
        caps, cap_len = self.get_caption(current_captions)
        return caps, cap_len, A_key[0:-4], A_cls_id

    def __len__(self):
        return len(self.dataset)
        # return 32