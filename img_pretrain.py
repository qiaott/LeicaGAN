from miscc.utils import mkdir_p, parse_str, save_pickle
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

def load_imgs(img_path, bbox=None):
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

    normalize = transforms.Compose([
                transforms.Resize((int(256 * 72 / 64), int(256 * 72 / 64))),
                transforms.RandomCrop(256),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    img = normalize(img)
    return img

def load_fakeimgs(img_path):
    import torchvision.transforms as transforms
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img = Image.open(img_path).convert('RGB')
    norm_img = normalize(img)
    return norm_img

def prepare_data(data):
    # index, captions, cap_lens = data
    index, captions, cap_lens, cls_id, imgs, fakeimgs = data
    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = torch.sort(cap_lens, 0, True)
    captions = captions[sorted_cap_indices].squeeze()
    imgs = imgs[sorted_cap_indices]
    fakeimgs = imgs[sorted_cap_indices]

    if cfg.CUDA:
        captions = Variable(captions).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
        imgs = Variable(imgs).cuda()
        fakeimgs = Variable(fakeimgs).cuda()
    else:
        captions = Variable(captions)
        sorted_cap_lens = Variable(sorted_cap_lens)
    return [index, captions, sorted_cap_lens, imgs, fakeimgs, cls_id]

class TextDataset(data.Dataset):
    def __init__(self, data_dir, base_size=64,
                 transform=None, target_transform=None):
        self.transform = transform
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
        bbox = img_data['bbox']
        key = img_data['img_name']
        cls_id = int(str(img_data['cls_index']))

        if cfg.DATASET_NAME == 'flower':
            img_name = '%s/images/%s.jpg' % (self.data_dir, key)
            key = key + '.jpg'
        else:
            img_name = '%s/images/%s/%s' % (self.data_dir, img_data['img_class'], key)
            fake_path = '%s/%s_EarlyGLAM/%s_Seg/Model/netG_epoch_%s/valid/single/%s_s-1.png' % (cfg.OUTPUT_DIR, cfg.DATASET_NAME, cfg.ENCODER1, cfg.RP.EPOCH, key[:-4])
        imgs = load_imgs(img_name, bbox)
        if os.path.isfile(fake_path):
            fakeimgs = load_fakeimgs(fake_path)
        else:
            fakeimgs = imgs
            print('No:', fake_path)
        return key, cls_id, bbox, img_name, imgs, fakeimgs

    def __getitem__(self, index): # index of text
        text_index = random.randint(0, self.embeddings_num)
        key, cls_id, bbox, name, imgs, fakeimgs = self.get_img_info(index)

        current_captions = self.dataset[index]['text'][text_index]
        current_caps, current_cap_len = self.get_caption(current_captions)
        # print(imgs.size(), fakeimgs.size())
        return index, current_caps, current_cap_len, cls_id, imgs, fakeimgs

    def __len__(self):
        return len(self.dataset)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/DAMSM/Ad_Class.yml', type=str)
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args

def evaluate(dataloader, cnn_model, rnn_model, batch_size):
    cnn_model.eval()
    rnn_model.eval()
    count = 0
    img2info = []
    for step, data in enumerate(dataloader, 0):
        index, captions, cap_lens, imgs, fake_imgs, cls_id = prepare_data(data)
        hidden = rnn_model.init_hidden(batch_size)
        _, sent_emb = rnn_model(captions, cap_lens, hidden)
        _, img_emb = cnn_model(imgs)
        _, fakeimg_emb = cnn_model(fake_imgs)
        # print(index.size())
        # print(sent_emb.size())
        index.data.cpu().numpy()
        cls_id.data.cpu().numpy()
        # sent_emb.data.cpu().numpy()
        img_emb.data.cpu().numpy()
        fakeimg_emb.data.cpu().numpy()
        for i in range(batch_size):
            imginfo = {}
            # print(index[i])
            imginfo['imgid'] = index[i].cpu().numpy()
            imginfo['clsid'] = cls_id[i].cpu().numpy()
            imginfo['real'] = img_emb[i].data.cpu().numpy()
            imginfo['fake'] = fakeimg_emb[i].data.cpu().numpy()
            imginfo['sent_emb'] = sent_emb[i].data.cpu().numpy()
            count += 1
            img2info.append(imginfo)
    print('processed %d data'%count)
    return img2info

def load_encoder(requires_grad_=False, attribute='', n_words=0):
    if attribute == 'image':
        current_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
        encoder_path = os.path.join(cfg.OUTPUT_DIR, cfg.DATASET_NAME+'_'+ cfg.ENCODER1, 'Model/image_encoder200.pth')
        print('load image encoder:', encoder_path)
    else:
        current_encoder = RNN_ENCODER(n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        encoder_path = os.path.join(cfg.OUTPUT_DIR, cfg.DATASET_NAME+'_'+ cfg.ENCODER1, 'Model/text_encoder200.pth')
        print('load text encoder:', encoder_path)
    state_dict = torch.load(encoder_path, map_location=lambda storage, loc: storage)
    current_encoder.load_state_dict(state_dict)
    for p in current_encoder.parameters():
        if requires_grad_:
            p.requires_grad = True
            print('Load an encoder from:', encoder_path, 'requires_grad is True')
        else:
            p.requires_grad = False
    current_encoder.eval()
    if cfg.CUDA:
        current_encoder.cuda()
    return current_encoder

def build_models(n_words):
    # load image encoder:
    image_encoder = load_encoder(False, 'image', n_words)
    # load 1st text encoder - semantic
    if cfg.ENCODER1 != '':
        text_encoder = load_encoder(False, 'text', n_words)
    else:  # load text encoder:
        print('Error: no pretrained text-image encoders')
        return

    return [text_encoder, image_encoder]

if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    cfg.GPU_ID = parse_str(cfg.GPU_ID)
    torch.cuda.set_device(cfg.GPU_ID[0])
    output_dir = '%s/%s_%s' % (cfg.OUTPUT_DIR, cfg.DATASET_NAME, cfg.ENCODER1)
    model_dir = os.path.join(output_dir, 'Model')
    image_dir = os.path.join(output_dir, 'Image')
    mkdir_p(model_dir)
    mkdir_p(image_dir)
    expect_image_encoder = os.path.join(model_dir, 'image_encoder200.pth')
    expect_text_encoder = os.path.join(model_dir, 'text_encoder200.pth')
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
    dataset = TextDataset(os.path.join(cfg.DATA_DIR,cfg.DATASET_NAME),
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=None)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, drop_last=True,
        shuffle=False, num_workers=int(cfg.WORKERS))
    text_encoder, image_encoder = build_models(dataset.n_words)
    img2info = evaluate(dataloader, image_encoder, text_encoder, batch_size)
    save_path = os.path.join(model_dir, 'img2info_list.pickle')
    save_pickle(img2info, save_path)