import random
import numpy as np
import torch
import re

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clean_text(text):

    text = re.sub(r"http\S+","",text)
    text = re.sub(r"@\w+","",text)
    text = text.strip()

    return text
