import torch
from model.net import Transformer
from transformers import AutoTokenizer
import argparse
from typing import List
import torch.nn as nn
from collections import OrderedDict
import numpy as np
from tqdm import tqdm

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

def beam_search(model, tok, args):
    model.eval()
    source = args.source

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
    sos_id = tok.vocab["<s>"]
    eos_id = tok.vocab["</s>"]

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
    pred=beam.buildTargetTokens(hyp)[:args.beam_size]                                                   # Best Prediction in Candidate list
    pred=[torch.cat([x.view(-1) for x in p]+[zero]*(args.max_len-len(p))).view(1,-1) for p in pred]
    preds.append(torch.cat(pred,0).unsqueeze(0))
    
    preds=torch.cat(preds,0)                
    return preds   
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, help="Input sequence to translate")
    parser.add_argument("--ckpt-path", type=str, default="checkpoints/version-2/best_model_step_56249_loss_1.2378.pt", help="Checkpoint path to decode")

    parser.add_argument("--n-enc-block", type=int, default=12)
    parser.add_argument("--n-dec-block", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--fc-hidden", type=int, default=2048)
    parser.add_argument(
        "--num-head", type=int, default=12, help="Number of self-attention head"
    )
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--beam_size", type=float, default=8)
    args = parser.parse_args()

    assert args.source, "You should enter source text to translate."
    assert args.ckpt_path, "You should enter trained checkpoint path."

    # load checkpoint
    tok = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    
    if hparams.do_ast:
        new_tokens = prepare_new_tokens()
        tok.add_tokens(new_tokens)
    
    model = Transformer(
        vocab_size=len(tok.vocab),
        num_enc_block=args.n_enc_block,
        num_dec_block=args.n_dec_block,
        num_head=args.num_head,
        hidden=args.hidden,
        fc_hidden=args.fc_hidden,
        dropout=args.dropout,
    )
    state_dict = torch.load(args.ckpt_path)
    new_state_dict = OrderedDict()
    
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.to(DEVICE)
    
    preds = beam_search(model, tok, args)

    for pred in preds:
        t=pred[0].cpu().numpy()
        t=list(t)
        if 0 in t:
            t=t[:t.index(0)]
        text = tok.decode(t,clean_up_tokenization_spaces=False)
        print(text)
