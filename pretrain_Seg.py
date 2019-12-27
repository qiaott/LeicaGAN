from miscc.utils import mkdir_p, build_super_images, parse_str
from miscc.losses import sent_loss, words_loss, MultiModalDLoss, ClassLoss
from cfg.config import cfg, cfg_from_file
from datasets import TextDataset, prepare_data
from model import RNN_ENCODER, CNN_ENCODER, DomainClassifier, Classifier
import os
from os import path
import sys
import time
import random
import pprint
import argparse
import numpy as np
from PIL import Image
import torch
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

UPDATE_INTERVAL = 20
def parse_args():
    parser = argparse.ArgumentParser(description='Train a Text-Visual co-Embedding Network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='', type=str)
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


def train(dataloader, cnn_model, rnn_model, domain_classifier, classifier, batch_size, \
          labels, img_domain_labels, text_domain_labels, optimizer, epoch, ixtoword, image_dir):
    cnn_model.train()
    rnn_model.train()
    domain_classifier.train()
    classifier.train()
    s_total_loss0 = 0
    s_total_loss1 = 0
    w_total_loss0 = 0
    w_total_loss1 = 0
    domain_total_loss = 0
    class_total_loss = 0
    count = (epoch + 1) * len(dataloader)
    for step, data in enumerate(dataloader, 0):
        rnn_model.zero_grad()
        cnn_model.zero_grad()
        domain_classifier.zero_grad()
        classifier.zero_grad()

        imgs, captions, cap_lens, \
            class_ids, keys = prepare_data(data)

        # region_features: batch_size x nef x 17 x 17
        # img_feature: batch_size x nef
        region_features, img_feature = cnn_model(imgs[-1])
        # --> batch_size x nef x 17*17
        # nef, att_sze = region_features.size(1), region_features.size(2)
        # region_features = region_features.view(batch_size, nef, -1)

        # hidden = rnn_model.module.init_hidden(batch_size)
        hidden = rnn_model.init_hidden(batch_size)
        # words_emb: batch_size x nef x seq_len
        # sent_emb: batch_size x nef
        words_emb, sent_emb = rnn_model(captions, cap_lens, hidden)

        # word-level loss:
        w_loss0, w_loss1, attn_maps = words_loss(region_features, words_emb, labels,
                                                 cap_lens, class_ids, batch_size)

        w_total_loss0 += w_loss0.data
        w_total_loss1 += w_loss1.data
        # loss = w_loss0 + w_loss1

        # sentence-level loss:
        s_loss0, s_loss1 = sent_loss(img_feature, sent_emb, labels, class_ids, batch_size)
        # loss += s_loss0 + s_loss1
        s_total_loss0 += s_loss0.data
        s_total_loss1 += s_loss1.data

        # domain loss:
        domain_loss = MultiModalDLoss(domain_classifier, img_feature, sent_emb,
                                      img_domain_labels, text_domain_labels)

        # domain loss:
        domain_total_loss += cfg.TRAIN.WEIGHT.LAMBDA2 * domain_loss.data

        new_class_ids = [x-1 for x in class_ids]
        cls_target = Variable(torch.LongTensor(new_class_ids)).cuda()
        class_loss = ClassLoss(classifier, img_feature, sent_emb, cls_target)
        class_total_loss += cfg.TRAIN.WEIGHT.LAMBDA3 * class_loss.data

        loss = cfg.TRAIN.WEIGHT.LAMBDA1 * (w_loss0 + w_loss1 + s_loss0 + s_loss1) \
               + cfg.TRAIN.WEIGHT.LAMBDA2 * domain_loss \
               + cfg.TRAIN.WEIGHT.LAMBDA3 * class_loss

        loss.backward()

        # `clip_grad_norm` helps prevent
        # the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(rnn_model.parameters(),
                                      cfg.TRAIN.RNN_GRAD_CLIP)
        optimizer.step()

        if step % UPDATE_INTERVAL == 0:
            count = epoch * len(dataloader) + step

            s_cur_loss0 = s_total_loss0.item() / UPDATE_INTERVAL
            s_cur_loss1 = s_total_loss1.item() / UPDATE_INTERVAL

            w_cur_loss0 = w_total_loss0.item() / UPDATE_INTERVAL
            w_cur_loss1 = w_total_loss1.item() / UPDATE_INTERVAL

            domain_cur_loss = domain_total_loss.item() / UPDATE_INTERVAL
            class_cur_loss = class_total_loss.item() / UPDATE_INTERVAL

            print(('| epoch {:3d} | {:5d}/{:5d} batches | '
                  's_loss {:5.2f}, {:5.2f} | '
                  'w_loss {:5.2f}, {:5.2f} | '
                  'domain {:5.2f} | '
                  'class {:5.2f} |'
                  .format(epoch, step, len(dataloader),
                          s_cur_loss0, s_cur_loss1,
                          w_cur_loss0, w_cur_loss1, domain_cur_loss, class_cur_loss)))

            training_message = '| epoch {:3d} | {:5d}/{:5d} batches | '\
                  's_loss {:5.2f}, {:5.2f} | ' 'w_loss {:5.2f}, {:5.2f} | ' 'domain {:5.2f} | ''class {:5.2f} |' \
                  .format(epoch, step, len(dataloader),
                          s_cur_loss0, s_cur_loss1,
                          w_cur_loss0, w_cur_loss1, domain_cur_loss, class_cur_loss)
            training_log_path = '%s/%s_%s/train_loss_log.txt' % (cfg.OUTPUT_DIR, cfg.DATASET_NAME, cfg.CONFIG_NAME)
            with open(training_log_path, "a") as log_file:
                log_file.write('%s\n' % training_message)  # save the training message

            s_total_loss0 = 0
            s_total_loss1 = 0
            w_total_loss0 = 0
            w_total_loss1 = 0
            domain_total_loss = 0
            class_total_loss = 0

            # attention Maps
            # img_set, _ = build_super_images(imgs[-1].cpu(), captions,
            #                        ixtoword, attn_maps, att_sze)
            # if img_set is not None:
            #     im = Image.fromarray(img_set)
            #     fullpath = '%s/attention_maps%d.png' % (image_dir, step)
            #     im.save(fullpath)
    return count


def evaluate(dataloader, cnn_model, rnn_model, domain_classifier, classifier, batch_size):
    cnn_model.eval()
    rnn_model.eval()
    domain_classifier.eval()
    classifier.eval()
    s_total_loss = 0
    w_total_loss = 0
    domain_total_loss = 0
    class_total_loss = 0
    for step, data in enumerate(dataloader, 0):
        real_imgs, captions, cap_lens, \
                class_ids, keys = prepare_data(data)

        region_features, img_feature = cnn_model(real_imgs[-1])
        # nef = region_features.size(1)
        # region_features = region_features.view(batch_size, nef, -1)

        hidden = rnn_model.init_hidden(batch_size)
        words_emb, sent_emb = rnn_model(captions, cap_lens, hidden)

        w_loss0, w_loss1, attn = words_loss(region_features, words_emb, labels,
                                            cap_lens, class_ids, batch_size)
        w_total_loss += (w_loss0 + w_loss1).data

        s_loss0, s_loss1 = sent_loss(img_feature, sent_emb, labels, class_ids, batch_size)
        s_total_loss += (s_loss0 + s_loss1).data

        domain_loss = MultiModalDLoss(domain_classifier, img_feature, sent_emb,
                                      img_domain_labels, text_domain_labels)
        domain_total_loss +=  cfg.TRAIN.WEIGHT.LAMBDA2 * domain_loss.data

        new_class_ids = [x - 1 for x in class_ids]
        cls_target = Variable(torch.LongTensor(new_class_ids)).cuda()
        class_loss = ClassLoss(classifier, img_feature, sent_emb, cls_target)
        class_total_loss += cfg.TRAIN.WEIGHT.LAMBDA3 * class_loss.data
        if step == 50:
            break

    s_cur_loss = s_total_loss.item() / (step*2)
    w_cur_loss = w_total_loss.item() / (step*2)
    domain_cur_loss = domain_total_loss.item() / step
    class_cur_loss = class_total_loss.item() / step

    return s_cur_loss, w_cur_loss, domain_cur_loss, class_cur_loss

def build_models(GPU_ID):
    # build model ############################################################
    text_encoder = RNN_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
    domain_classifier = DomainClassifier()
    classifier = Classifier()
    labels = Variable(torch.LongTensor(list(range(batch_size))))

    img_domain_labels = Variable((torch.cat((torch.zeros(batch_size), torch.ones(batch_size)), 0)).view(batch_size, -1))
    text_domain_labels = Variable((torch.cat((torch.ones(batch_size), torch.zeros(batch_size)), 0)).view(batch_size, -1))
    start_epoch = 0
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
        # text_encoder = torch.nn.DataParallel(text_encoder, GPU_ID)  # multi-GPUs
        # image_encoder = torch.nn.DataParallel(image_encoder, GPU_ID)  # multi-GPUs
        domain_classifier.cuda()
        classifier.cuda()
        labels = labels.cuda()
        img_domain_labels = img_domain_labels.cuda()
        text_domain_labels = text_domain_labels.cuda()

    return text_encoder, image_encoder, domain_classifier, classifier, labels, img_domain_labels, text_domain_labels, start_epoch

if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    cfg.GPU_ID = parse_str(cfg.GPU_ID)
    torch.cuda.set_device(cfg.GPU_ID[0])
    output_dir = '%s/%s_%s' % \
        (cfg.OUTPUT_DIR, cfg.DATASET_NAME, cfg.CONFIG_NAME)
    model_dir = os.path.join(output_dir, 'Model')
    image_dir = os.path.join(output_dir, 'Image')
    mkdir_p(model_dir)
    mkdir_p(image_dir)

    expect_model = os.path.join(model_dir, 'image_encoder200.pth')
    if path.exists(expect_model):
        print('-' * 100)
        print('Congrats! The %s encoders already exist, the path is %s.' % (cfg.CONFIG_NAME, model_dir))
        print('Let us go to the next part!')
        print('-' * 100)
    else:
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
        # Get data loader ##################################################
        imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM-1))
        batch_size = cfg.TRAIN.BATCH_SIZE
        image_transform = transforms.Compose([
            transforms.Resize((int(imsize * 72 / 64), int(imsize * 72 / 64))),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip()])
        
        dataset = TextDataset(os.path.join(cfg.DATA_DIR,cfg.DATASET_NAME), 'train',
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform)
        print('dataset.n_words, dataset.embeddings_num:', dataset.n_words, dataset.embeddings_num)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=True,
            shuffle=True, num_workers=int(cfg.WORKERS))

        # # validation data #
        dataset_val = TextDataset(os.path.join(cfg.DATA_DIR,cfg.DATASET_NAME), 'test',
                                  base_size=cfg.TREE.BASE_SIZE,
                                  transform=image_transform)
        dataloader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=batch_size, drop_last=True,
            shuffle=True, num_workers=int(cfg.WORKERS))

        # Train ##############################################################
        text_encoder, image_encoder, domain_classifier, classifier, labels, img_domain_labels, text_domain_labels, start_epoch = build_models(cfg.GPU_ID)

        para = list(text_encoder.parameters())
        for v in image_encoder.parameters():
            if v.requires_grad:
                para.append(v)
        for v in domain_classifier.parameters():
            if v.requires_grad:
                para.append(v)
        for v in classifier.parameters():
            if v.requires_grad:
                para.append(v)

        # optimizer = optim.Adam(para, lr=cfg.TRAIN.ENCODER_LR, betas=(0.5, 0.999))
        # At any point you can hit Ctrl + C to break out of training early.
        try:
            lr = cfg.TRAIN.ENCODER_LR
            for epoch in range(start_epoch, cfg.TRAIN.MAX_EPOCH):
                optimizer = optim.Adam(para, lr, betas=(0.5, 0.999))
                epoch_start_time = time.time()

                count = train(dataloader, image_encoder, text_encoder, domain_classifier, classifier, \
                              batch_size, labels, img_domain_labels, text_domain_labels, optimizer, epoch,\
                              dataset.ixtoword, image_dir)
                print('-' * 100)
                if len(dataloader_val) > 0:
                    s_loss, w_loss, domain_loss, class_loss = evaluate(dataloader, image_encoder, text_encoder, domain_classifier, classifier, batch_size)
                    print(('|epoch {:3d} | test loss: |'
                          's_loss_avg {:5.2f} w_loss_avg {:5.2f} domain {:5.2f} class {:5.2f} | lr {:.5f}|'
                          .format(epoch, s_loss, w_loss, domain_loss, class_loss, lr)))

                # saving loss log
                testing_message = '|epoch {:3d} | test loss: |' \
                          's_loss_avg {:5.2f} w_loss_avg {:5.2f} domain {:5.2f} class {:5.2f} | lr {:.5f}|' \
                          .format(epoch, s_loss, w_loss, domain_loss, class_loss, lr)
                testing_log_path = '%s/%s_%s/test_loss_log.txt' % (cfg.OUTPUT_DIR, cfg.DATASET_NAME, cfg.CONFIG_NAME)
                with open(testing_log_path, "a") as log_file:
                    log_file.write('%s\n' % testing_message)  # save the testing message

                print('-' * 100)
                if lr > cfg.TRAIN.ENCODER_LR/10.:
                    lr *= 0.98

                if (epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0 or
                    epoch == cfg.TRAIN.MAX_EPOCH):
                    torch.save(image_encoder.state_dict(),
                               '%s/image_encoder%d.pth' % (model_dir, epoch))
                    torch.save(text_encoder.state_dict(),
                               '%s/text_encoder%d.pth' % (model_dir, epoch))
                    if cfg.TRAIN.WEIGHT.LAMBDA2:
                        torch.save(domain_classifier.state_dict(),
                               '%s/domain_classifier%d.pth' % (model_dir, epoch))
                    if cfg.TRAIN.WEIGHT.LAMBDA3:
                        torch.save(classifier.state_dict(),
                               '%s/classifier%d.pth' % (model_dir, epoch))
                    print('Save G/Ds/domain/classifier models.')
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')
