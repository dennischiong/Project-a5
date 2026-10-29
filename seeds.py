"""Fixed random seeds for every script in this repo."""
import os
import random

import numpy as np


def set_seed(seed: int = 0) -> int:
    """Seed python, numpy and (if installed) torch. Returns the seed used."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed
