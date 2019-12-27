from six.moves import range
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
from PIL import Image
from miscc.utils import print_para_nums
from cfg.config import cfg
from miscc.utils import mkdir_p, parse_str
from miscc.utils import build_super_images, build_super_images2, build_super_images3
from miscc.utils import weights_init, load_params, copy_G_params
from model import EarlyGLAM_G_NET
from model import D_NET64, D_NET128, D_NET256
from datasets import prepare_data
from model import RNN_ENCODER, CNN_ENCODER
from miscc.losses import words_loss, discriminator_loss, generator_loss, KL_loss
import os
import time
import numpy as np


class condGANTrainer(object):
    def __init__(self, output_dir, data_loader, n_words, ixtoword):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)

        cfg.GPU_ID = parse_str(cfg.GPU_ID)
        torch.cuda.set_device(cfg.GPU_ID[0])

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.n_words = n_words
        self.ixtoword = ixtoword
        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)

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
        # load image encoder:
        image_encoder = self.load_encoder(cfg.ENCODER1, False, 'image')

        # load 1st text encoder - semantic
        if cfg.ENCODER1 != '':
            text_encoder = self.load_encoder(cfg.ENCODER1, False, 'text')
        else: # load text encoder:
            print('Error: no pretrained text-image encoders')
            return

        # load 2nd text encoder - segmentation
        if cfg.ENCODER2 != '':
            seg_encoder = self.load_encoder(cfg.ENCODER2, False, 'text')
        else:
            print('No segmentation encoders..')
            seg_encoder = None

        # Generator:
        if cfg.GAN.GNET == 'EarlyGLAM':
            netG = EarlyGLAM_G_NET()
        else:
            print('No generator assigned...')
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
            netG.load_state_dict(state_dict, strict=False)
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
                    netsD[i].load_state_dict(state_dict, strict=False)

        if cfg.CUDA:
            netG = nn.DataParallel(netG, device_ids=cfg.GPU_ID)
            netsD = [nn.DataParallel(netD, device_ids=cfg.GPU_ID) for netD in netsD]
            netG.cuda()
            netsD = [netD.cuda() for netD in netsD]

        print_para_nums(image_encoder, 'image_encoder', only_trainale=False)
        print_para_nums(text_encoder, 'text_encoder', only_trainale=False)
        print_para_nums(seg_encoder, 'seg_encoder', only_trainale=False)
        print_para_nums(netG, cfg.GAN.GNET, only_trainale=False)
        [print_para_nums(netsD[i], 'Discriminator', only_trainale=False) for i in range(len(netsD))]

        return [text_encoder, seg_encoder, image_encoder, netG, netsD, epoch]

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
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (self.model_dir, epoch))
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
        fake_imgs, attention_maps, _, _, _, _ = netG(noise, sent_emb1, words_embs1, sent_emb2, words_embs2, mask)
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

    def train(self):
        text_encoder, seg_encoder, image_encoder, netG, netsD, start_epoch = self.build_models()

        avg_param_G = copy_G_params(netG)
        optimizerG, optimizersD = self.define_optimizers(netG, netsD)
        real_labels, fake_labels, match_labels = self.prepare_labels()

        batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz))
        fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))
        if cfg.CUDA:
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        gen_iterations = 0
        # gen_iterations = start_epoch * self.num_batches
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()
            data_iter = iter(self.data_loader)
            step = 0
            while step < self.num_batches:
                ######################################################
                data = next(data_iter)
                imgs, captions, cap_lens, class_ids, keys = prepare_data(data)

                # encoder text using pretrained text-image matching model, MODEL1
                hidden = text_encoder.init_hidden(batch_size)
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                words_embs1, sent_emb1 = text_encoder(captions, cap_lens, hidden)
                words_embs1, sent_emb1 = words_embs1.detach(), sent_emb1.detach()

                # ENCODER2
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

                # (2) Generate fake images
                noise.data.normal_(0, 1)
                # fake_imgs, att_maps_w1, att_maps_w2, att_maps_s, mu, logvar
                if cfg.GAN.GNET == 'EarlyGLAM':
                    fake_imgs, _, _, _, mu, logvar = netG(noise, sent_emb1, words_embs1, sent_emb2, words_embs2, mask)
                else:
                    fake_imgs, _, _, _ = netG(noise, sent_emb1, words_embs1, sent_emb2, words_embs2, mask)
                # (3) Update D network
                errD_total = 0
                D_logs = ''
                for i in range(len(netsD)):
                    netsD[i].zero_grad()
                    errD = discriminator_loss(netsD[i], imgs[i], fake_imgs[i], sent_emb1, real_labels, fake_labels)
                    # backward and update parameters
                    errD.backward()
                    optimizersD[i].step()
                    errD_total += errD
                    D_logs += 'errD%d: %.2f ' % (i, errD.item())

                step += 1
                gen_iterations += 1

                netG.zero_grad()
                errG_total, G_logs = \
                    generator_loss(netsD, image_encoder, fake_imgs, real_labels,
                                   words_embs1, sent_emb1, match_labels, cap_lens, class_ids)
                kl_loss = KL_loss(mu, logvar)
                errG_total += kl_loss
                G_logs += 'kl_loss: %.2f ' % kl_loss.item()
                # backward and update parameters
                errG_total.backward()
                optimizerG.step()
                for p, avg_p in zip(netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                if gen_iterations % 100 == 0:
                    print((D_logs + '\n' + G_logs))
                # save images
                if gen_iterations % 1000 == 0:
                    backup_para = copy_G_params(netG)
                    load_params(netG, avg_param_G)
                    self.save_img_results(netG, fixed_noise, sent_emb1, words_embs1, sent_emb2, \
                                          words_embs2, mask, image_encoder, \
                                          captions, cap_lens, epoch, name='average')
                    load_params(netG, backup_para)
            end_t = time.time()

            print(('''[%d/%d][%d]
                  Loss_D: %.2f Loss_G: %.2f Time: %.2fs'''
                  % (epoch, self.max_epoch, self.num_batches,
                     errD_total.item(), errG_total.item(),
                     end_t - start_t)))

            if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:  # and epoch != 0:
                self.save_model(netG, avg_param_G, netsD, epoch)

        self.save_model(netG, avg_param_G, netsD, self.max_epoch)

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
            img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(fullpath)

    # sampling
    def sampling(self, split_dir):
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
        # load sencond text encoder, if it exists
        if cfg.ENCODER2 != '':
            seg_encoder = self.load_encoder(cfg.ENCODER2, False, 'text')
        else:
            print('no segmentation model used')
            seg_encoder = None

        batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
        noise = noise.cuda()

        model_dir = cfg.TRAIN.NET_G
        print('Load model:', model_dir)
        netG.load_state_dict(torch.load(model_dir), strict=False)
        s_tmp = model_dir[:model_dir.rfind('.pth')]
        save_dir = '%s/%s' % (s_tmp, split_dir)
        mkdir_p(save_dir)
        cnt = 0
        for _ in range(1):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
            for step, data in enumerate(self.data_loader, 0):
                cnt += batch_size
                if step % 100 == 0:
                    print('step: ', step)
                # if step > 50:
                #     break
                imgs, captions, cap_lens, class_ids, keys = prepare_data(data)
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
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
                if cfg.GAN.GNET == 'EarlyGLAM':
                    fake_imgs, _, _, _, _, _ = netG(noise, sent_emb1, words_embs1, sent_emb2, words_embs2, mask)
                else:
                    fake_imgs, _, _, _ = netG(noise, sent_emb1, words_embs1, sent_emb2, words_embs2, mask)
                for j in range(batch_size):
                    s_tmp = '%s/single/%s' % (save_dir, keys[j])
                    folder = s_tmp[:s_tmp.rfind('/')]
                    if not os.path.isdir(folder):
                        print('Make a new folder: ', folder)
                        mkdir_p(folder)
                    k = -1
                    # for k in range(len(fake_imgs)):
                    im = fake_imgs[k][j].data.cpu().numpy()
                    # [-1, 1] --> [0, 255]
                    im = (im + 1.0) * 127.5
                    im = im.astype(np.uint8)
                    im = np.transpose(im, (1, 2, 0))
                    im = Image.fromarray(im)
                    fullpath = '%s_s%d.png' % (s_tmp, k)
                    im.save(fullpath)

    # customed generation
    def gen_example(self, data_dic):
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
        # load sencond text encoder, if it exists
        if cfg.ENCODER2 != '':
            seg_encoder = self.load_encoder(cfg.ENCODER2, False, 'text')
        else:
            print('No segmentation model used..')
            seg_encoder = None

        model_dir = cfg.TRAIN.NET_G
        print('Load model:', model_dir)
        netG.load_state_dict(torch.load(model_dir), strict=False)

        save_dir = os.path.join(cfg.OUTPUT_DIR, 'smooth',
                             '%s_netG_epoch_%s' % (cfg.CONFIG_NAME, i))
        mkdir_p(save_dir)
        for key in data_dic:
            captions, cap_lens, sorted_indices = data_dic[key]
            batch_size = captions.shape[0]
            nz = cfg.GAN.Z_DIM
            captions = Variable(torch.from_numpy(captions), volatile=True)
            cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)
            captions = captions.cuda()
            cap_lens = cap_lens.cuda()
            for i in range(1):  # 16
                noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
                noise = noise.cuda()
                # (1) Extract text embeddings
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
                # G attention
                for j in range(batch_size):
                    save_name = '%s/%d_s_%d' % (save_dir, i, sorted_indices[j])
                    for k in range(len(fake_imgs)):
                        im = fake_imgs[k][j].data.cpu().numpy()
                        im = (im + 1.0) * 127.5
                        im = im.astype(np.uint8)
                        # print('im', im.shape)
                        im = np.transpose(im, (1, 2, 0))
                        # print('im', im.shape)
                        im = Image.fromarray(im)
                        fullpath = '%s_g%d.png' % (save_name, k)
                        im.save(fullpath)
                        print('save image to path:', fullpath)



