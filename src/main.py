import glob
import os

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoTokenizer

from config import load_config
from dataset import get_loader
from model.net import Transformer
from trainer import Trainer
from utils import ResultWriter, fix_seed


def main(rank, hparams, ngpus_per_node: int):
    fix_seed(hparams.seed)
    resultwriter = ResultWriter(hparams.result_path)
    if hparams.distributed:
        hparams.rank = hparams.rank * ngpus_per_node + rank
        print(f"Use GPU {hparams.rank} for training")
        dist.init_process_group(
            backend=hparams.dist_backend,
            init_method=hparams.dist_url,
            world_size=hparams.world_size,
            rank=hparams.rank,
        )

    # get shared tokenizer and vocab
    if hparams.distributed:
        if rank != 0:
            dist.barrier()
        tok = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        if rank == 0:
            dist.barrier()
    else:
        tok = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    # get dataloaders
    loaders = [
        get_loader(
            tok=tok,
            batch_size=hparams.batch_size,
            root_path=hparams.root_path,
            workers=hparams.workers,
            max_len=hparams.max_len,
            mode=mode,
            distributed=hparams.distributed,
        )
        for mode in ["train", "valid"]
    ]

    # get model and initialize
    model = Transformer(
        vocab_size=len(tok.vocab),
        num_enc_block=hparams.n_enc_block,
        num_dec_block=hparams.n_dec_block,
        num_head=hparams.num_head,
        hidden=hparams.hidden,
        fc_hidden=hparams.fc_hidden,
        dropout=hparams.dropout,
    )
    
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)

    # training phase
    trainer = Trainer(hparams, loaders, model, resultwriter, pad_idx=tok.pad_token_id)
    best_result = trainer.fit()

    # testing phase
    if rank in [-1, 0]:
        version = best_result["version"]
        state_dict = torch.load(
            glob.glob(
                os.path.join(hparams.ckpt_path, f"version-{version}/best_model_*.pt")
            )[0],
        )
        test_loader = get_loader(
            tok=tok,
            batch_size=hparams.batch_size,
            root_path=hparams.root_path,
            workers=hparams.workers,
            max_len=hparams.max_len,
            mode="test",
        )
        test_result = trainer.test(test_loader, state_dict)

        # save result
        best_result.update(test_result)
        resultwriter.update(hparams, **best_result)


if __name__ == "__main__":
    hparams = load_config()
    ngpus_per_node = torch.cuda.device_count()

    if hparams.distributed:
        hparams.rank = 0
        hparams.world_size = ngpus_per_node * hparams.world_size
        mp.spawn(main, nprocs=ngpus_per_node, args=(hparams, ngpus_per_node))
    else:
        main(hparams.rank, hparams, ngpus_per_node)
