import torch
import torch.nn as nn
from cfg.config import cfg

def conv1x1(in_planes, out_planes):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=False)

def linear(input_dim, output_dim):
    return nn.Linear(input_dim, output_dim).cuda()


def func_attention(query, region_feature, gamma1):
    """
    query: batch x nef x queryL
    region_feature: batch x nef x ih x iw (sourceL=ihxiw)
    mask: batch_size x sourceL
    """
    batch_size, queryL = query.size(0), query.size(2)
    ih, iw = region_feature.size(2), region_feature.size(3)
    sourceL = ih * iw

    # --> batch x sourceL x nef
    region_feature = region_feature.view(batch_size, -1, sourceL)
    contextT = torch.transpose(region_feature, 1, 2).contiguous()

    # Get attention
    # -->batch x sourceL x queryL
    att = torch.bmm(contextT, query)
    # --> batch*sourceL x queryL
    att = att.view(batch_size*sourceL, queryL)
    att = nn.Softmax(dim=1)(att)
    # --> batch x sourceL x queryL
    att = att.view(batch_size, sourceL, queryL)
    # --> batch*queryL x sourceL
    att = torch.transpose(att, 1, 2).contiguous()
    att = att.view(batch_size*queryL, sourceL)
    att = nn.Softmax(dim=1)(att * gamma1)
    att = att.view(batch_size, queryL, sourceL)
    # --> batch x sourceL x queryL
    attT = torch.transpose(att, 1, 2).contiguous()

    # (batch x nef x sourceL)(batch x sourceL x queryL)
    # --> batch x nef x queryL
    weightedContext = torch.bmm(region_feature, attT)

    return weightedContext, att.view(batch_size, -1, ih, iw)


class GlobalAttentionGeneral(nn.Module):
    """
    self.att = ATT_NET(ngf, nef)
    """
    def __init__(self, ngf, nef):
        super(GlobalAttentionGeneral, self).__init__()
        self.conv_context = conv1x1(nef, ngf)
        self.sm = nn.Softmax(1)
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask  # batch x sourceL

    def forward(self, input, context):
        """
            input: batch x ngf x ih x iw (queryL=ihxiw)
            context: batch x nef x sourceL
        """
        ih, iw = input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context.size(0), context.size(2)

        # --> batch x queryL x ngf
        target = input.view(batch_size, -1, queryL)
        targetT = torch.transpose(target, 1, 2).contiguous()
        # batch x nef x sourceL --> batch x nef x sourceL x 1
        sourceT = context.unsqueeze(3)
        # --> batch x ngf x sourceL
        sourceT = self.conv_context(sourceT).squeeze(3)

        # Get attention
        # (batch x queryL x ngf)(batch x ngf x sourceL)
        # -->batch x queryL x sourceL
        att = torch.bmm(targetT, sourceT)
        # --> batch*queryL x sourceL
        att = att.view(batch_size*queryL, sourceL)
        if self.mask is not None:
            # batch_size x sourceL --> batch_size*queryL x sourceL
            mask = self.mask.repeat(queryL, 1)
            att.data.masked_fill_(mask.data, -float('inf'))
        att = self.sm(att)
        # --> batch x queryL x sourceL
        att = att.view(batch_size, queryL, sourceL)
        # --> batch x sourceL x queryL
        att = torch.transpose(att, 1, 2).contiguous()
        weightedContext = torch.bmm(sourceT, att)
        weightedContext = weightedContext.view(batch_size, -1, ih, iw)
        att = att.view(batch_size, -1, ih, iw)
        return weightedContext, att


class EarlyGLAMGeneral(nn.Module):
    def __init__(self, idf, cdf):
        super(EarlyGLAMGeneral, self).__init__()
        self.conv_context = conv1x1(cdf, idf)
        self.conv_sentence_vis = conv1x1(idf, idf)
        self.linear = nn.Linear(100, idf)
        self.sm = nn.Softmax()
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask  # batch x sourceL

    def att_blcok(self, input, context, ih, iw, queryL, batch_size, sourceL):
        # # generated image feature:--> batch x queryL x idf

        target = input.view(batch_size, -1, queryL)             # batch x idf x queryL
        targetT = torch.transpose(target, 1, 2).contiguous()    # batch x queryL x idf
        sourceT = context.unsqueeze(3)
        # --> batch x idf x sourceL
        sourceT = self.conv_context(sourceT).squeeze(3)

        # Get attention
        attn = torch.bmm(targetT, sourceT)
        # --> batch*queryL x sourceL
        attn = attn.view(batch_size*queryL, sourceL)
        if self.mask is not None:
            # batch_size x sourceL --> batch_size*queryL x sourceL
            mask = self.mask.repeat(queryL, 1)
            attn.data.masked_fill_(mask.data, -float('inf'))
        attn = self.sm(attn)  # Eq. (2)
        # --> batch x queryL x sourceL
        attn = attn.view(batch_size, queryL, sourceL)
        # --> batch x sourceL x queryL
        attn = torch.transpose(attn, 1, 2).contiguous()
        # (batch x idf x sourceL)(batch x sourceL x queryL)
        # --> batch x idf x queryL
        weightedContext = torch.bmm(sourceT, attn)
        # weightedContext = weightedContext.view(batch_size, -1, ih, iw)  # batch x idf x ih x iw
        word_attn = attn.view(batch_size, -1, ih, iw)  # (batch x sourceL x ih x iw)
        return weightedContext, word_attn

    def fusion_block(self, weightedContext1, weightedContext2, batch_size, ih, iw):
        # weightedContext1/weightedContext2: batch x idf x queryL
        weightedContext = (1 - cfg.TRAIN.BALANCE) *  weightedContext1 + cfg.TRAIN.BALANCE * weightedContext2
        weightedContext = weightedContext.view(batch_size, -1, ih, iw)  # batch x idf x ih x iw
        return weightedContext

    def forward(self, input, sentence, context1, context2):
        idf, ih, iw = input.size(1), input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context1.size(0), context1.size(2)

        weightedContext1, word_attn1 = self.att_blcok(input, context1, ih, iw, queryL, batch_size, sourceL)

        if context2 is not None:
            weightedContext2, word_attn2 = self.att_blcok(input, context2, ih, iw, queryL, batch_size, sourceL)
            weightedContext = self.fusion_block(weightedContext1, weightedContext2, batch_size, ih, iw)
        else:
            weightedContext = weightedContext1

        sentence = self.linear(sentence)
        sentence = sentence.view(batch_size, idf, 1, 1)
        sentence = sentence.repeat(1, 1, ih, iw)
        sentence_vs = torch.mul(input, sentence)   # batch x idf x ih x iw
        sentence_vs = self.conv_sentence_vis(sentence_vs) # batch x idf x ih x iw
        sent_att = nn.Softmax()(sentence_vs)  # batch x idf x ih x iw
        weightedSentence = torch.mul(sentence, sent_att)  # batch x idf x ih x iw

        return weightedContext, weightedSentence, word_attn1, word_attn2, sent_att