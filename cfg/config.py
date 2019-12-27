import os.path as osp
import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Dataset name: flower, birds
__C.DATA_DIR = ''
__C.OUTPUT_DIR = ''
__C.DATASET_NAME = ''
__C.CONFIG_NAME = ''
__C.ENCODER1 = ''
__C.ENCODER2 = ''
__C.ENCODER3 = ''
__C.SPLIT = ''
__C.GPU_ID = '1,0'
__C.CUDA = True
__C.WORKERS = 4
__C.MODE = ''
__C.RNN_TYPE = 'LSTM'   # 'GRU'
__C.B_VALIDATION = False

__C.TREE = edict()
__C.TREE.BRANCH_NUM = 3
__C.TREE.BASE_SIZE = 64

# Training options
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 64
__C.TRAIN.MAX_EPOCH = 600
__C.TRAIN.SNAPSHOT_INTERVAL = 2000
__C.TRAIN.DISCRIMINATOR_LR = 2e-4
__C.TRAIN.GENERATOR_LR = 2e-4
__C.TRAIN.ENCODER_LR = 2e-4
__C.TRAIN.RNN_GRAD_CLIP = 0.25
__C.TRAIN.FLAG = True
__C.TRAIN.B_NET_D = True
__C.TRAIN.NET_G = ''
__C.TRAIN.NET_E = ''

# weights for the pretrained image-text matching models:
__C.TRAIN.WEIGHT = edict()
__C.TRAIN.WEIGHT.LAMBDA1 = 1.0 # weight of word-sentence similarity loss
__C.TRAIN.WEIGHT.LAMBDA2 = 0.0 # weight of domain loss
__C.TRAIN.WEIGHT.LAMBDA3 = 0.0 # weight of class loss
__C.TRAIN.WEIGHT.LAMBDA4 = 0.0 # weight for triplet loss

__C.TRAIN.SMOOTH = edict()
__C.TRAIN.SMOOTH.GAMMA1 = 5.0  # smooth of
__C.TRAIN.SMOOTH.GAMMA2 = 5.0  # smooth of
__C.TRAIN.SMOOTH.GAMMA3 = 10.0 # smooth of
__C.TRAIN.SMOOTH.LAMBDA = 1.0  # smooth of
__C.TRAIN.SMOOTH.LAMBDA2 = 0.0  # smooth of

__C.TRAIN.BALANCE = 0.2

# Modal options
__C.GAN = edict()
__C.GAN.DF_DIM = 64
__C.GAN.GF_DIM = 128
__C.GAN.Z_DIM = 100
__C.GAN.CONDITION_DIM = 100
__C.GAN.R_NUM = 2
__C.GAN.B_ATTENTION = True
__C.GAN.GNET = ''


__C.TEXT = edict()
__C.TEXT.CAPTIONS_PER_IMAGE = 10
__C.TEXT.EMBEDDING_DIM = 256
__C.TEXT.WORDS_NUM = 18

#Layout options - LO
__C.LO = edict()
__C.LO.IMG_DIM=2048
__C.LO.MARGIN = 0.2
__C.LO.MAX_VIOLATION = True  #max_violation
__C.LO.LAMBDA_SOFTMAX = 4.0  #lambda_softmax
__C.LO.LAMBDA_LSE = 6.0      #lambda_lse
__C.LO.AGG_FUNC = 'Mean'     #agg_func
__C.LO.RAW_FEATURE_NORM = 'clipped_l2norm' #raw_feature_norm
__C.LO.CROSS_ATTN = 'i2t'    #cross_attn
__C.LO.LR_UPDATE = 10        #lr_update
__C.LO.LEARNING_RATE = 0.002 # learning rate

__C.RP = edict()
__C.RP.EPOCH = 400

__C.ATT = edict()
__C.ATT.VISUALIZATION = False

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in list(a.items()):
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print(('Error under config key: {}'.format(k)))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)
