import os
import random

import numpy as np
import torch
import warnings


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Avoid forcing CUDA initialization on CPU-only / misconfigured environments.
    # `torch.cuda.is_available()` can emit a warning if the driver/runtime is broken;
    # we suppress that specific warning so CPU runs stay clean.
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"CUDA initialization: CUDA unknown error.*",
                category=UserWarning,
            )
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
    except Exception:
        # If CUDA is absent/broken, keep going on CPU.
        pass

    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
