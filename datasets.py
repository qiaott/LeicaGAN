#coding=utf-8
import os
from cfg.config import cfg
import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import numpy.random as random
import pickle

from miscc.utils import load_pickle

def prepare_data(data):
    imgs, captions, captions_lens, class_ids, keys = data

    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(captions_lens, 0, True)

    real_imgs = []
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_cap_indices]
        if cfg.CUDA:
            real_imgs.append(Variable(imgs[i]).cuda())
        else:
            real_imgs.append(Variable(imgs[i]))

    captions = captions[sorted_cap_indices].squeeze()
    class_ids = class_ids[sorted_cap_indices].numpy()
    keys = [keys[i] for i in sorted_cap_indices.numpy()]
    if cfg.CUDA:
        captions = Variable(captions).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
    else:
        captions = Variable(captions)
        sorted_cap_lens = Variable(sorted_cap_lens)

    return [real_imgs, captions, sorted_cap_lens,
            class_ids, keys]

def prepare_tii_data(data):
    A1_imgs, A2_imgs, B1_imgs, captions, captions_lens, A1_cls_id, B1_cls_id, A1_keys = data
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(captions_lens, 0, True)

    A1_real_imgs = []
    A2_real_imgs = []
    B1_real_imgs = []
    for i in range(len(A1_imgs)):
        A1_imgs[i] = A1_imgs[i][sorted_cap_indices]
        if cfg.CUDA:
            A1_real_imgs.append(Variable(A1_imgs[i]).cuda())
        else:
            A1_real_imgs.append(Variable(A1_imgs[i]))
    for i in range(len(A2_imgs)):
        A2_imgs[i] = A2_imgs[i][sorted_cap_indices]
        if cfg.CUDA:
            A2_real_imgs.append(Variable(A2_imgs[i]).cuda())
        else:
            A2_real_imgs.append(Variable(A2_imgs[i]))
    for i in range(len(B1_imgs)):
        B1_imgs[i] = B1_imgs[i][sorted_cap_indices]
        if cfg.CUDA:
            B1_real_imgs.append(Variable(B1_imgs[i]).cuda())
        else:
            B1_real_imgs.append(Variable(B1_imgs[i]))

    captions = captions[sorted_cap_indices].squeeze()
    A1_cls_id = A1_cls_id[sorted_cap_indices].numpy()
    A1_keys = [A1_keys[i] for i in sorted_cap_indices.numpy()]
    if cfg.CUDA:
        captions = Variable(captions).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
    else:
        captions = Variable(captions)
        sorted_cap_lens = Variable(sorted_cap_lens)

    return [A1_real_imgs, A2_real_imgs, B1_real_imgs, captions, sorted_cap_lens,
            A1_cls_id, B1_cls_id, A1_keys]


def load_imgs(img_path, imsize, bbox=None, transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    ret = []
    for i in range(cfg.TREE.BRANCH_NUM):
        if i < (cfg.TREE.BRANCH_NUM - 1):
            re_img = transforms.Scale(imsize[i])(img)
        else:
            re_img = img
        ret.append(normalize(re_img))
    return ret

def get_diff_img(current_cls_id, cls2imgid):
    all_cls = list(cls2imgid.keys())
    cls_p = (np.array(all_cls) != current_cls_id) / (len(all_cls) - 1)
    diff_cls_idx = np.random.choice(np.array(all_cls), 1, p=cls_p)[0]
    img_idx = np.random.choice(cls2imgid[diff_cls_idx], 1)[0]
    return diff_cls_idx, img_idx

# get another image from same class
def get_similar_cls_image(current_img_idx, idx_opts):
    # get the other random image from same class of image1
    img_p = (np.array(idx_opts) != current_img_idx) / (len(idx_opts) - 1)
    img_idx = np.random.choice(idx_opts, 1, p=img_p)[0]
    assert img_idx != current_img_idx
    return img_idx


class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train',
                 base_size=64,
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
        if cfg.CONFIG_NAME == 'Layout':
            layout_path = os.path.join(cfg.DATA_DIR, cfg.DATASET_NAME, 'hard_feature.pickle')
            with open(layout_path, 'rb') as f:
                self.hard_atts = pickle.load(f, encoding='iso-8859-1')
        self.ixtoword, self.wordtoix, self.n_words = self.load_dictionary(data_dir+'/dictionary.pickle')
        if split=='train':
            if cfg.SPLIT == '0': # '0' is old split as in AttnGAN and StackGAN
                dataset_path = self.data_dir + '/train_dataset_ub.pickle'
                print('load the old split')
            else:  # this is the new split proposed in the new work
                print('load the new split')
                dataset_path = self.data_dir + '/train_dataset.pickle'
        elif split=='test':
            if cfg.SPLIT == '0': # '0' is old split as in AttnGAN and StackGAN
                dataset_path = self.data_dir + '/test_dataset_ub.pickle'
            else:  # this is the new split proposed in the new work
                dataset_path = self.data_dir + '/test_dataset.pickle'
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

    def __getitem__(self, index):
        current_data = self.dataset[index]
        key = current_data['img_name']
        cls_id = int(str(current_data['cls_index']))

        if current_data['bbox']:
            bbox = current_data['bbox']
        else:
            bbox = None

        if cfg.CONFIG_NAME == 'Seg':
            if cfg.DATASET_NAME == 'bird':
                img_name = ('%s/segmentation/%s/%s' % (self.data_dir, current_data['img_class'], key)).replace('jpg', 'png')
                imgs = load_imgs(img_name, self.imsize,
                            bbox, self.transform, normalize=self.norm)
            elif cfg.DATASET_NAME == 'flower':
                img_name = ('%s/segmentation/%s.jpg' % (self.data_dir, key.replace('image', 'segmim')))
                imgs = load_imgs(img_name, self.imsize,
                            bbox, self.transform, normalize=self.norm)
        elif cfg.DATASET_NAME == 'flower':
            img_name = '%s/images/%s.jpg' % (self.data_dir, key)
            imgs = load_imgs(img_name, self.imsize,
                             bbox, self.transform, normalize=self.norm)
            key = key+'.jpg'
        else:
            img_name = '%s/images/%s/%s' % (self.data_dir, current_data['img_class'], key)
            imgs = load_imgs(img_name, self.imsize,
                            bbox, self.transform, normalize=self.norm)

        # random select a sentence
        random_ix = random.randint(0, self.embeddings_num)
        current_captions = current_data['text'][random_ix]
        caps, cap_len = self.get_caption(current_captions)
        return imgs, caps, cap_len, cls_id, key[0:-4]

    def __len__(self):
        return len(self.dataset)

class TIIDataset(data.Dataset):
    def __init__(self, data_dir, split='train',
                 base_size=64,
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
        if split=='train':
            if cfg.SPLIT == '0': # '0' is old split as in AttnGAN and StackGAN
                dataset_path = self.data_dir + '/train_dataset_up.pickle'
            else:  # this is the new split proposed in the new work
                dataset_path = self.data_dir + '/train_dataset.pickle'
            self.cls2imgid = load_pickle(data_dir + '/train_cls2imgid.pickle')
        elif split=='test':
            if cfg.SPLIT == '0': # '0' is old split as in AttnGAN and StackGAN
                dataset_path = self.data_dir + '/test_dataset_up.pickle'
            else:  # this is the new split proposed in the new work
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
        bbox = img_data['bbox'] if img_data['bbox'] else None
        if cfg.CONFIG_NAME == 'Seg':
            if cfg.DATASET_NAME == 'bird':
                img_name = ('%s/segmentation/%s/%s' % (self.data_dir, img_data['img_class'], key)).replace('jpg', 'png')
            elif cfg.DATASET_NAME == 'flower':
                img_name = ('%s/segmentation/%s.jpg' % (self.data_dir, key)).replace('image', 'segmim')
        elif cfg.DATASET_NAME == 'flower':
            img_name = '%s/images/%s.jpg' % (self.data_dir, key)
        else:
            img_name = '%s/images/%s/%s' % (self.data_dir, img_data['img_class'], key)
        imgs = load_imgs(img_name, self.imsize, bbox, self.transform, normalize=self.norm)
        return key, cls_id, bbox, img_name, imgs

    def __getitem__(self, A1_index):
        # get A1: data, key, cls_id, bbox
        A1_key, A1_cls_id, A1_bbox, A1_name, A1_imgs = self.get_img_info(A1_index)

        # random select a sentence for A1
        random_ix = random.randint(0, self.embeddings_num)
        current_captions = self.dataset[A1_index]['text'][random_ix]
        caps, cap_len = self.get_caption(current_captions)
        idx_opts = self.cls2imgid[A1_cls_id]
        A2_index = get_similar_cls_image(A1_index, idx_opts)
        A2_key, A2_cls_id, A2_bbox, A2_name, A2_imgs = self.get_img_info(A2_index)

        # get B1: data, key, cls_id, bbox
        B1_cls_id, B1_index = get_diff_img(A1_cls_id, self.cls2imgid)
        B1_key, B1_cls_id, B1_bbox, B1_name, B1_imgs = self.get_img_info(B1_index)

        return A1_imgs, A2_imgs, B1_imgs, caps, cap_len, A1_cls_id, B1_cls_id, A1_key[0:-4]

    def __len__(self):
        return len(self.dataset)



