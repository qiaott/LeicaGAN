import scipy
from six.moves import range
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
from PIL import Image
from cfg.config import cfg
from miscc.utils import mkdir_p, parse_str
from miscc.utils import build_super_images
from miscc.utils import weights_init, load_params, copy_G_params
from model import EarlyGLAM_G_NET
from model import D_NET64, D_NET128, D_NET256
from rp_dataset import prepare_RP_data as prepare_data
from model import RNN_ENCODER, CNN_ENCODER
from miscc.losses import words_loss
import os
from miscc.utils import load_pickle
import numpy as np
from rp_dataset import get_mis_99

class condGANTrainer(object):
    def __init__(self, output_dir, data_loader, n_words, ixtoword):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)

        cfg.GPU_ID = parse_str(cfg.GPU_ID)
        torch.cuda.set_device(cfg.GPU_ID[0])
        cudnn.benchmark = True

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.n_words = n_words
        self.ixtoword = ixtoword
        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)
        data_dir = os.path.join(cfg.DATA_DIR, cfg.DATASET_NAME)
        self.cls2imgid = load_pickle(data_dir + '/test_cls2imgid.pickle')
        encoder_dir = '%s/%s_%s/Model' % (cfg.OUTPUT_DIR, cfg.DATASET_NAME, cfg.ENCODER1)
        self.img2textfeature = load_pickle(os.path.join(encoder_dir, 'testimg2textfeature.pickle'))


    def load_encoder(self, encoder, requires_grad_=False, attribute=''):
        if attribute == 'image':
            current_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
            encoder_path = os.path.join(cfg.OUTPUT_DIR, cfg.DATASET_NAME+'_'+ encoder, 'Model/image_encoder200.pth')
        else:
            current_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            encoder_path = os.path.join(cfg.OUTPUT_DIR, cfg.DATASET_NAME+'_'+ encoder, 'Model/text_encoder200.pth')
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

    def build_models(self):

        image_encoder = self.load_encoder(cfg.ENCODER1, False, 'image')

        if cfg.ENCODER1 != '':
            text_encoder = self.load_encoder(cfg.ENCODER1, False, 'text')
        else: # load text encoder:
            print('Error: no pretrained text-image encoders')
            return

        if cfg.ENCODER2 != '':
            seg_encoder = self.load_encoder(cfg.ENCODER2, False, 'text')
        else:
            print('No segmentation model used..')
            seg_encoder = None

        # Generator:
        if cfg.GAN.GNET == 'EarlyGLAM':
            netG = EarlyGLAM_G_NET()
        else:
            print('no generator assigned.')
        netG.apply(weights_init)

        # Discriminator:
        netsD = []
        if cfg.TREE.BRANCH_NUM > 0:
            netsD.append(D_NET64())
        if cfg.TREE.BRANCH_NUM > 1:
            netsD.append(D_NET128())
        if cfg.TREE.BRANCH_NUM > 2:
            netsD.append(D_NET256())
        for i in range(len(netsD)):
            netsD[i].apply(weights_init)
        print('# of netsD', len(netsD))

        epoch = 0
        if cfg.TRAIN.NET_G != '':
            state_dict = \
                torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', cfg.TRAIN.NET_G)
            istart = cfg.TRAIN.NET_G.rfind('_') + 1
            iend = cfg.TRAIN.NET_G.rfind('.')
            epoch = cfg.TRAIN.NET_G[istart:iend]
            epoch = int(epoch) + 1
            if cfg.TRAIN.B_NET_D:
                Gname = cfg.TRAIN.NET_G
                for i in range(len(netsD)):
                    s_tmp = Gname[:Gname.rfind('/')]
                    Dname = '%s/netD%d.pth' % (s_tmp, i)
                    print('Load D from: ', Dname)
                    state_dict = \
                        torch.load(Dname, map_location=lambda storage, loc: storage)
                    netsD[i].load_state_dict(state_dict)

        # G/D uses CUDA
        if cfg.CUDA:
            netG = nn.DataParallel(netG, device_ids=cfg.GPU_ID)
            netsD = [nn.DataParallel(netD, device_ids=cfg.GPU_ID) for netD in netsD]
            netG.cuda()
            netsD = [netD.cuda() for netD in netsD]

        return [text_encoder, seg_encoder, layout_encoder, image_encoder, netG, netsD, epoch]

    def define_optimizers(self, netG, netsD):

        optimizersD = [optim.Adam(netD.parameters(),
                                  lr=cfg.TRAIN.DISCRIMINATOR_LR,
                                  betas=(0.5, 0.999)) for netD in netsD]
        optimizerG = optim.Adam(netG.parameters(),
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(0.5, 0.999))
        return optimizerG, optimizersD

    def prepare_labels(self):
        batch_size = self.batch_size
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        match_labels = Variable(torch.LongTensor(list(range(batch_size))))
        if cfg.CUDA:
            real_labels = real_labels.cuda()
            fake_labels = fake_labels.cuda()
            match_labels = match_labels.cuda()
        return real_labels, fake_labels, match_labels

    def save_model(self, netG, avg_param_G, netsD, epoch):
        backup_para = copy_G_params(netG)
        load_params(netG, avg_param_G)
        torch.save(netG.state_dict(),
            '%s/netG_epoch_%d.pth' % (self.model_dir, epoch))
        load_params(netG, backup_para)

        for i in range(len(netsD)):
            netD = netsD[i]
            torch.save(netD.state_dict(), '%s/netD%d.pth' % (self.model_dir, i))
        print('Save G/Ds models.')

    def set_requires_grad_value(self, models_list, brequires):
        for i in range(len(models_list)):
            for p in models_list[i].parameters():
                p.requires_grad = brequires

    def save_img_results(self, netG, noise, sent_emb1, words_embs1, sent_emb2, words_embs2, mask,
                         image_encoder, captions, cap_lens,
                         gen_iterations, name='current'):
        # Save images
        fake_imgs, attention_maps, _, _ = netG(noise, sent_emb1, words_embs1, sent_emb2, words_embs2, mask)
        for i in range(len(attention_maps)):
            if len(fake_imgs) > 1:
                img = fake_imgs[i + 1].detach().cpu()
                lr_img = fake_imgs[i].detach().cpu()
            else:
                img = fake_imgs[0].detach().cpu()
                lr_img = None
            attn_maps = attention_maps[i]
            att_sze = attn_maps.size(2)
            img_set, _ = build_super_images(img, captions, self.ixtoword,
                                   attn_maps, att_sze, lr_imgs=lr_img)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/G_%s_%d_%d.png'%(self.image_dir, name, gen_iterations, i)
                im.save(fullpath)

        i = -1
        img = fake_imgs[i].detach()
        region_features, _ = image_encoder(img)
        att_sze = region_features.size(2)
        _, _, att_maps = words_loss(region_features.detach(),
                                    words_embs1.detach(),
                                    None, cap_lens,
                                    None, self.batch_size)
        img_set, _ = \
            build_super_images(fake_imgs[i].detach().cpu(),
                               captions, self.ixtoword, att_maps, att_sze)
        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = '%s/D_%s_%d.png'\
                % (self.image_dir, name, gen_iterations)
            im.save(fullpath)

    def save_singleimages(self, images, filenames, save_dir,
                          split_dir, sentenceID=0):
        for i in range(images.size(0)):
            s_tmp = '%s/single_samples/%s/%s' %\
                (save_dir, split_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)

            fullpath = '%s_%d.jpg' % (s_tmp, sentenceID)
            # range from [-1, 1] to [0, 1]
            # img = (images[i] + 1.0) / 2
            img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            # range from [0, 1] to [0, 255]
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(fullpath)

    def load_imgs(self, img_path):
        import torchvision.transforms as transforms
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        img = Image.open(img_path).convert('RGB')
        norm_img = normalize(img)
        return norm_img

    def R_precision(self, split_dir):
        # Generator:
        if cfg.GAN.GNET == 'EarlyGLAM':
            netG = EarlyGLAM_G_NET()
        else:
            print('no generator assigned.')
        netG.cuda()
        netG = nn.DataParallel(netG, device_ids=cfg.GPU_ID)
        netG.apply(weights_init)
        netG.eval()
        # load text encoder:
        text_encoder = self.load_encoder(cfg.ENCODER1, False, 'text')
        image_encoder = self.load_encoder(cfg.ENCODER1, False, 'image')
        # load sencond text encoder, if it exists
        if cfg.ENCODER2 != '':
            seg_encoder = self.load_encoder(cfg.ENCODER2, False, 'text')
        else:
            print('No segmentation model used..')
            seg_encoder = None

        batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
        noise = noise.cuda()

        rp_results = [0, 0, 0, 0, 0]
        rp_count = 0
        model_dir = cfg.TRAIN.NET_G
        print('Load model:', model_dir)
        netG.load_state_dict(torch.load(model_dir), strict=False)
        s_tmp = model_dir[:model_dir.rfind('.pth')]
        save_dir = '%s/%s' % (s_tmp, split_dir)
        mkdir_p(save_dir)
        cnt = 0
        for _ in range(1):
            for step, data in enumerate(self.data_loader, 0):
                cnt += batch_size
                if step % 100 == 0:
                    print('step: ', step)
                captions, cap_lens, keys, cls_id = prepare_data(data)
                hidden = text_encoder.init_hidden(batch_size)
                words_embs1, sent_emb1 = text_encoder(captions, cap_lens, hidden)
                words_embs1, sent_emb1 = words_embs1.detach(), sent_emb1.detach()
                if seg_encoder is not None:
                    seg_hidden = seg_encoder.init_hidden(batch_size)
                    words_embs2, sent_emb2 = text_encoder(captions, cap_lens, seg_hidden)
                    words_embs2, sent_emb2 = words_embs2.detach(), sent_emb2.detach()
                else:
                    words_embs2, sent_emb2 = None, None

                mask = (captions == 0)
                num_words = words_embs1.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]
                # Generate fake images
                noise.data.normal_(0, 1)
                fake_imgs, _, _, _, _, _ = netG(noise, sent_emb1, words_embs1, sent_emb2, words_embs2, mask)
                _, img_feature = image_encoder(fake_imgs[0])
                for j in range(batch_size):
                    rp_count += 1
                    random_texts = get_mis_99(cls_id[j].item(), self.cls2imgid, self.img2textfeature)
                    rp_result = self.calculate_rp(img_feature[j].detach().cpu(), sent_emb1[j].detach().cpu(),
                                                  random_texts)
                    rp_results = np.sum([rp_results, rp_result], axis=0)
            # model_path = os.path.join(cfg.OUTPUT_DIR, cfg.DATASET_NAME + '_' + cfg.GAN.GNET, cfg.CONFIG_NAME, 'Model')
            rp1 = round(rp_results[0] / rp_count, 4)
            rp2 = round((rp_results[0] + rp_results[1]) / rp_count, 4)
            rp3 = round((rp_results[0] + rp_results[1] + rp_results[2])/ rp_count, 4)
            rp4 = round((rp_results[0] + rp_results[1] + rp_results[2] + rp_results[3]) / rp_count, 4)
            rp5 = round((rp_results[0] + rp_results[1] + rp_results[2] + rp_results[3] + rp_results[4])/ rp_count, 4)

            content = 'Recall top1-top5: %s,%s,%s,%s,%s' % (rp1,rp2,rp3,rp4,rp5)
            # self.write2txt(i, content, model_path)
            print(content)

    def calculate_rp(self, img_feature, sent_emb, random_texts):
        score = []
        # print(img_feature)
        score_match = scipy.spatial.distance.cosine(img_feature, sent_emb)
        score.append(score_match)
        # print('len(random_texts):', len(random_texts))
        for random_text in random_texts:
            score_mismatch = scipy.spatial.distance.cosine(img_feature, random_text)
            score.append(score_mismatch)
        # print(len(score))
        result = np.argsort(score)
        top1 = top2 = top3 = top4 = top5 = 0
        if result[0] == 0:
            top1 = 1
        if result[1] == 0:
            top2 = 1
        if result[2] == 0:
            top3 = 1
        if result[3] == 0:
            top4 = 1
        if result[4] == 0:
            top5 = 1
        return top1, top2, top3, top4, top5






