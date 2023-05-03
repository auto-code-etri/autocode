import os
import math
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


def fix_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ResultWriter:
    def __init__(self, directory):

        self.dir = directory
        self.hparams = None
        self.load()
        self.writer = dict()

    def update(self, args, **results):
        now = datetime.now()
        date = "%s-%s %s:%s" % (now.month, now.day, now.hour, now.minute)
        self.writer.update({"date": date})
        self.writer.update(results)
        self.writer.update(vars(args))

        if self.hparams is None:
            self.hparams = pd.DataFrame(self.writer, index=[0])
        else:
            self.hparams = self.hparams.append(self.writer, ignore_index=True)
        self.save()

    def save(self):
        assert self.hparams is not None
        self.hparams.to_csv(self.dir, index=False)

    def load(self):
        path = os.path.split(self.dir)[0]
        if not os.path.exists(path):
            os.makedirs(path)
            self.hparams = None
        elif os.path.exists(self.dir):
            self.hparams = pd.read_csv(self.dir)
        else:
            self.hparams = None


class SequentialDistributedSampler(Sampler):
    """
    Distributed Sampler that subsamples indices sequentially, making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training), which means that the model params won't
    have to be synced (i.e. will not hang for synchronization even if varied number of forward passes), we still add
    extra samples to the sampler to make it evenly divisible (like in `DistributedSampler`) to make it easy to `gather`
    or `reduce` resulting tensors at the end of the loop.

    Ref: https://github.com/huggingface/transformers/blob/6bef764506b2b53732ee5315b80f89e6c007b584/src/transformers/trainer_pt_utils.py#L185
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert (
            len(indices) == self.total_size
        ), f"Indices length {len(indices)} and total size {self.total_size} mismatched"

        # subsample
        indices = indices[
            self.rank * self.num_samples : (self.rank + 1) * self.num_samples
        ]
        assert (
            len(indices) == self.num_samples
        ), f"Indices length {len(indices)} and sample number {self.num_samples} mismatched"

        return iter(indices)

    def __len__(self):
        return self.num_samples
