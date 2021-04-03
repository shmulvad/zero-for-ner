import random

import numpy as np
import torch

GLOVE = 'glove'
CONCEPTNET = 'conceptnet'
COMBINED = 'combined'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_whitespace(c):
    return c in [" ", "\t", "\r", "\n"] or ord(c) == 0x202F
