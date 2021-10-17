import sys
import os
sys.path.append(os.path.dirname('/lustre/S/fuqiang/unilm/unilm/unilm-v1/'))

import logging
import glob
import math
import json
import argparse
import random
from pathlib import Path
from tqdm import tqdm, trange
import numpy as np

import torch
from torch.utils.data import RandomSampler, SequentialSampler, BatchSampler
from torch.utils.data.distributed import DistributedSampler

from src.pytorch_pretrained_bert.tokenization import BertTokenizer as BertTokenizert
from src.pytorch_pretrained_bert.tokenization import WhitespaceTokenizer as WhitespaceTokenizert
# from src.pytorch_pretrained_bert.modeling import BertForPreTrainingLossMask
# from src.pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

# from torch.nn.data_parallel import DataParallelImbalance
import src.biunilm.seq2seq_loader as seq2seq_loadert
import torch.distributed as dist

from reprod_log import ReprodLogger, ReprodDiffHelper

log_torch = ReprodLogger()


parser = argparse.ArgumentParser()

parser.add_argument("--data_dir",
                    default=None,
                    type=str,
                    required=True,
                    help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
parser.add_argument("--src_file", default=None, type=str,
                    help="The input data file name.")
parser.add_argument("--tgt_file", default=None, type=str,
                    help="The output data file name.")
parser.add_argument("--bert_model", default=None, type=str, required=True,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                         "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
parser.add_argument("--config_path", default=None, type=str,
                    help="Bert config file path.")
parser.add_argument("--output_dir",
                    default=None,
                    type=str,
                    # required=True,
                    help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--log_dir",
                    default='',
                    type=str,
                    # required=True,
                    help="The output directory where the log will be written.")
parser.add_argument("--model_recover_path",
                    default=None,
                    type=str,
                    # required=True,
                    help="The file of fine-tuned pretraining model.")
parser.add_argument("--optim_recover_path",
                    default=None,
                    type=str,
                    help="The file of pretraining optimizer.")

# Other parameters
parser.add_argument("--max_seq_length",
                    default=128,
                    type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                         "Sequences longer than this will be truncated, and sequences shorter \n"
                         "than this will be padded.")
parser.add_argument("--do_train",
                    action='store_true',
                    help="Whether to run training.")
parser.add_argument("--do_eval",
                    action='store_true',
                    help="Whether to run eval on the dev set.")
parser.add_argument("--do_lower_case",
                    action='store_true',
                    help="Set this flag if you are using an uncased model.")
parser.add_argument("--train_batch_size",
                    default=32,
                    type=int,
                    help="Total batch size for training.")
parser.add_argument("--eval_batch_size",
                    default=64,
                    type=int,
                    help="Total batch size for eval.")
parser.add_argument("--learning_rate", default=5e-5, type=float, 
                    help="The initial learning rate for Adam.")
parser.add_argument("--label_smoothing", default=0, type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay",
                    default=0.01,
                    type=float,
                    help="The weight decay rate for Adam.")
parser.add_argument("--finetune_decay",
                    action='store_true',
                    help="Weight decay to the original weights.")
parser.add_argument("--num_train_epochs",
                    default=3.0,
                    type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion",
                    default=0.1,
                    type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                         "E.g., 0.1 = 10%% of training.")
parser.add_argument("--hidden_dropout_prob", default=0.1, type=float,
                    help="Dropout rate for hidden states.")
parser.add_argument("--attention_probs_dropout_prob", default=0.1, type=float,
                    help="Dropout rate for attention probabilities.")
parser.add_argument("--no_cuda",
                    action='store_true',
                    help="Whether not to use CUDA when available")
parser.add_argument("--local_rank",
                    type=int,
                    default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="random seed for initialization")
parser.add_argument('--gradient_accumulation_steps',
                    type=int,
                    default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument('--fp16', action='store_true',
                    help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument('--fp32_embedding', action='store_true',
                    help="Whether to use 32-bit float precision instead of 16-bit for embeddings")
parser.add_argument('--loss_scale', type=float, default=0,
                    help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                         "0 (default value): dynamic loss scaling.\n"
                         "Positive power of 2: static loss scaling value.\n")
parser.add_argument('--amp', action='store_true',
                    help="Whether to use amp for fp16")
parser.add_argument('--from_scratch', action='store_true',
                    help="Initialize parameters with random values (i.e., training from scratch).")
parser.add_argument('--new_segment_ids', action='store_true',
                    help="Use new segment ids for bi-uni-directional LM.")
parser.add_argument('--new_pos_ids', action='store_true',
                    help="Use new position ids for LMs.")
parser.add_argument('--tokenized_input', action='store_true',
                    help="Whether the input is tokenized.")
parser.add_argument('--max_len_a', type=int, default=0,
                    help="Truncate_config: maximum length of segment A.")
parser.add_argument('--max_len_b', type=int, default=0,
                    help="Truncate_config: maximum length of segment B.")
parser.add_argument('--trunc_seg', default='',
                    help="Truncate_config: first truncate segment A/B (option: a, b).")
parser.add_argument('--always_truncate_tail', action='store_true',
                    help="Truncate_config: Whether we should always truncate tail.")
parser.add_argument("--mask_prob", default=0.15, type=float,
                    help="Number of prediction is sometimes less than max_pred when sequence is short.")
parser.add_argument("--mask_prob_eos", default=0, type=float,
                    help="Number of prediction is sometimes less than max_pred when sequence is short.")
parser.add_argument('--max_pred', type=int, default=20,
                    help="Max tokens of prediction.")
parser.add_argument("--num_workers", default=0, type=int,
                    help="Number of workers for the data loader.")

parser.add_argument('--mask_source_words', action='store_true',
                    help="Whether to mask source words for training")
parser.add_argument('--skipgram_prb', type=float, default=0.0,
                    help='prob of ngram mask')
parser.add_argument('--skipgram_size', type=int, default=1,
                    help='the max size of ngram mask')
parser.add_argument('--mask_whole_word', action='store_true',
                    help="Whether masking a whole word.")
parser.add_argument('--do_l2r_training', action='store_true',
                    help="Whether to do left to right training")
parser.add_argument('--has_sentence_oracle', action='store_true',
                    help="Whether to have sentence level oracle for training. "
                         "Only useful for summary generation")
parser.add_argument('--max_position_embeddings', type=int, default=None,
                    help="max position embeddings")
parser.add_argument('--relax_projection', action='store_true',
                    help="Use different projection layers for tasks.")
parser.add_argument('--ffn_type', default=0, type=int,
                    help="0: default mlp; 1: W((Wx+b) elem_prod x);")
parser.add_argument('--num_qkv', default=0, type=int,
                    help="Number of different <Q,K,V>.")
parser.add_argument('--seg_emb', action='store_true',
                    help="Using segment embedding for self-attention.")
parser.add_argument('--s2s_special_token', action='store_true',
                    help="New special tokens ([S2S_SEP]/[S2S_CLS]) of S2S.")
parser.add_argument('--s2s_add_segment', action='store_true',
                    help="Additional segmental for the encoder of S2S.")
parser.add_argument('--s2s_share_segment', action='store_true',
                    help="Sharing segment embeddings for the encoder of S2S (used with --s2s_add_segment).")
parser.add_argument('--pos_shift', action='store_true',
                    help="Using position shift for fine-tuning.")

args = parser.parse_args()

def build_torch_data_pipeline():
     fn_src = os.path.join(
               args.data_dir, args.src_file if args.src_file else 'test.src')
     fn_tgt = os.path.join(
               args.data_dir, args.tgt_file if args.tgt_file else 'test.tgt')

     tokenizer = BertTokenizert.from_pretrained(
          args.bert_model, do_lower_case=args.do_lower_case)

     if args.max_position_embeddings:
          tokenizer.max_len = args.max_position_embeddings
     data_tokenizer = WhitespaceTokenizert() if args.tokenized_input else tokenizer

     print("Loading Train Dataset", args.data_dir)
     bi_uni_pipeline = [seq2seq_loadert.Preprocess4Seq2seq(args.max_pred, args.mask_prob, list(tokenizer.vocab.keys(
          )), tokenizer.convert_tokens_to_ids, args.max_seq_length, new_segment_ids=args.new_segment_ids, truncate_config={'max_len_a': args.max_len_a, 'max_len_b': args.max_len_b, 'trunc_seg': args.trunc_seg, 'always_truncate_tail': args.always_truncate_tail}, mask_source_words=args.mask_source_words, skipgram_prb=args.skipgram_prb, skipgram_size=args.skipgram_size, mask_whole_word=args.mask_whole_word, mode="s2s", has_oracle=args.has_sentence_oracle, num_qkv=args.num_qkv, s2s_special_token=args.s2s_special_token, s2s_add_segment=args.s2s_add_segment, s2s_share_segment=args.s2s_share_segment, pos_shift=args.pos_shift)]
     file_oracle = None
     test_dataset = seq2seq_loadert.Seq2SeqDataset(
               fn_src, fn_tgt, args.train_batch_size, data_tokenizer, args.max_seq_length, file_oracle=file_oracle, bi_uni_pipeline=bi_uni_pipeline)

     test_sampler = torch.utils.data.SequentialSampler(test_dataset)
     _batch_size = args.train_batch_size
     # test_batch_sampler = torch.utils.data.BatchSampler(sampler=test_sampler, batch_size=_batch_size, drop_last=False)
     test_dataloader = torch.utils.data.DataLoader(test_dataset, sampler=test_sampler, batch_size = _batch_size, shuffle = False,
                                                  num_workers=args.num_workers, collate_fn=seq2seq_loadert.batch_list_to_batch_tensors, pin_memory=True)
     return test_dataset, test_dataloader



if __name__ =="__main__":
     torch_dataset, torch_dataloader = build_torch_data_pipeline()
     log_torch.add("length", np.array(len(torch_dataset)))
     
     for idx in range(5):
          log_torch.add(f"dataset_{idx}",
                         np.array(torch_dataset[idx][0]))
          
     for idx,  torch_batch in enumerate(torch_dataloader):
          if idx >= 5:
               break
          log_torch.add(f"dataloader_{idx}",
                              torch_batch[0].detach().cpu().numpy())
     
     log_torch.save("torch_data.npy")
     

     
 