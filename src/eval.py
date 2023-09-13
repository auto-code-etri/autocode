import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, T5ForConditionalGeneration
from datasets import load_dataset
from typing import List
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import json
import jsonlines
from config import load_config
from dataset import prepare_new_tokens
from model.net import *

import os
import logging
from bleu import _bleu

from human_eval.data import write_jsonl, read_problems
import subprocess

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEVICE = "cuda"

class Beam(object):
    def __init__(self, size,sos,eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                       .fill_(0)]
        self.nextYs[0][0] = sos

        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.
        Parameters:
        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step
        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))


        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >=self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished=[]
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i)) 
            unfinished.sort(key=lambda a: -a[0])
            self.finished+=unfinished[:self.size-len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps=[]
        for _,timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j+1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps
    
    def buildTargetTokens(self, preds):
        sentence=[]
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok==self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence

def add_pad(tok, max_len: int, indice: List[int]) -> List[int]:
    diff = max_len - len(indice)
    if diff > 0:
        indice += [tok.pad_token_id] * diff
    else:
        indice = indice[:max_len]
    assert len(indice) == max_len
    return indice

def get_src_mask(tok, indice: torch.Tensor) -> torch.Tensor:
    return (indice.data == tok.pad_token_id).unsqueeze(-2)

def get_tgt_mask(tok, indice: torch.Tensor) -> torch.Tensor:
    mask = (indice.data != tok.pad_token_id).unsqueeze(-2)
    mask = mask & subsequent_mask(indice.shape[-1]).type_as(mask.data)
    return ~mask

def subsequent_mask(size: int) -> torch.Tensor:
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(subsequent_mask) == 0

def beam_search(model, tok, args, so):
    model.eval()
    source = so

    source_ids = torch.tensor(
        add_pad(tok, args.max_len, tok.encode(source, add_special_tokens=False)),
        device=torch.device(DEVICE),
    )
    source_ids = source_ids.unsqueeze(0)
    source_mask = get_src_mask(tok, source_ids)
    outputs = model.encoder(model.src_embed(source_ids), source_mask)
    encoder_output = outputs.repeat(args.beam_size, 1, 1)                                               # [args.beam_size, max_length, hidden_dim]
    source_mask = source_mask.repeat(args.beam_size, 1, 1)                                              # [args.beam_size, max_length, hidden_dim]

    lsm = nn.LogSoftmax(dim=-1).to(DEVICE)
    preds=[]
    sos_id = tok.bos_token
    eos_id = tok.eos_token

    zero=torch.cuda.LongTensor(1).fill_(0)  
    beam = Beam(args.beam_size,sos_id,eos_id)
    input_ids=beam.getCurrentState()                                                                    # [args.beam_size, generated_token_size]

    for _ in tqdm(range(args.max_len)):
        if beam.done():
            break
        attn_mask= get_tgt_mask(tok, input_ids).unsqueeze(1)                                            # [args.beam_size, 1, generated_token_size, generated_token_size]
        out = model.decoder(model.tgt_embed(input_ids), encoder_output, source_mask, attn_mask)
        out = lsm(model.generator(out)[:,-1,:]).data                                                    # [args.beam_size, vocab_size]
        beam.advance(out)
        input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
        input_ids=torch.cat((input_ids,beam.getCurrentState()),-1)
        
    hyp= beam.getHyp(beam.getFinal())
    pred=beam.buildTargetTokens(hyp)[:args.beam_size]
    pred=[torch.cat([x.view(-1) for x in p]+[zero]*(args.max_len-len(p))).view(1,-1) for p in pred]
    preds.append(torch.cat(pred,0).unsqueeze(0))
    
    preds=torch.cat(preds,0)                
    return preds   

def bleu(tok):
    preds = open("result.txt", "r").readlines()
    gts = open("answers.json", "r").readlines()

    assert len(preds) == len(gts), f"Samples of predictions and answers are not equal, {len(preds)}: {len(gts)}"

    total = len(gts)
    EM = 0.0
    with open("ground_truth.txt", "w") as wf:
        for pred, gt in zip(preds, gts):
            pred = pred.strip()
            gt = json.loads(gt)["code"]
            wf.write(gt+"\n")
            if pred.split() == gt.split():
                EM += 1

    bleu_score = round(_bleu("ground_truth.txt", "result.txt", tok), 2)
    logger.info(f"BLEU: {bleu_score}, EM: {round(EM/total*100, 2)}")
    print(f"BLEU: {bleu_score}, EM: {round(EM/total*100, 2)}")

    try:
        os.remove("ground_truth.txt")
    except Exception:
        pass

if __name__ == "__main__":

    # load checkpoint
    config = AutoConfig.from_pretrained("Salesforce/codegen2-1B")
    tok = AutoTokenizer.from_pretrained("Salesforce/codegen2-1B")
    model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen2-1B", trust_remote_code=True, revision="main")
    model.to(DEVICE)

    #args = load_config()
    #tok = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    #if args.do_ast:
    #    new_tokens = prepare_new_tokens()
    #    tok.add_tokens(new_tokens)
#
    #model = Transformer(
    #    vocab_size=len(tok.vocab),
    #    num_enc_block=args.n_enc_block,
    #    num_dec_block=args.n_dec_block,
    #    num_head=args.num_head,
    #    hidden=args.hidden,
    #    fc_hidden=args.fc_hidden,
    #    dropout=args.dropout,
    #)
    #path = ""
    #model = torch.load(path)


    lines = []

    if not os.path.exists("./result.txt") and not os.path.exists("./answers.json"):
        with open("./eval_data/concode/test.json", "r") as f:
            lines += f.readlines()
        with open("result.txt", "w") as f, jsonlines.open("answers.json", "w", flush=True) as f1:
            for num, line in enumerate(tqdm(lines)):
                code = line[10:line.find("}\"")+1]
                nl = line[line.find("nl\":")+6:line.find("}\n")-1]
                raw = {'code': code, 'nl': nl,}
                f1.write(raw)
                inputs = tok(nl, return_tensors="pt").input_ids.to(DEVICE)
                sample = model.generate(inputs, num_beams=12, max_length=256 + len(inputs))
                result = tok.decode(sample[0], skip_special_tokens=True)
                f.write(result.replace("\n", "\\n") + "\n")
                f.flush()
            bleu(tok)
    else:
        bleu(tok)

    problems = read_problems()

    num_samples_per_task = 100
    with jsonlines.open("samples.jsonl", "w", flush=True) as f1:
        for _ in tqdm(range(num_samples_per_task)):
            for task_id in problems:
                inputs = tok(problems[task_id]["prompt"], return_tensors="pt", max_length=127).to(DEVICE)
                sample = model.generate(**inputs, num_beams=12, max_length=128, pad_token_id=tok.eos_token_id)
                result = tok.decode(sample[0])
                f1.write(dict(task_id=task_id, completion=result[len(problems[task_id]["prompt"]):]))
    #write_jsonl("samples.jsonl", samples)

    humaneval_result = subprocess.run("evaluate_functional_correctness samples.jsonl", shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    print(humaneval_result.stdout.decode("utf-8"))
