import torch
import random
import os
import numpy as np

class config:
    batch_size = 4
    stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    monet_dir = '../input/gan-getting-started/monet_jpg/'
    photo_dir = '../input/gan-getting-started/photo_jpg/'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True