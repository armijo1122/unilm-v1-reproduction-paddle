import sys
import os

from paddle.optimizer import AdamW
sys.path.append(os.path.dirname('/lustre/S/fuqiang/unilm/unilm/unilm-v1/'))

import random
import numpy as np
from functools import partial
from paddlenlp.data import Stack, Tuple, Pad
from utils.data_processor import convert_example, create_dataloader

import paddle
import paddlenlp
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import LinearDecayWithWarmup

import logging
import glob
import math
import json
import argparse
import random
from pathlib import Path
from tqdm import tqdm, trange
import numpy as np
from paddle.io import RandomSampler
from paddle.io import DistributedBatchSampler
from paddle.optimizer import Adamax

from src_paddle.pytorch_pretrained_bert.tokenization import BertTokenizer, WhitespaceTokenizer
from src_paddle.pytorch_pretrained_bert.modeling import BertForPreTrainingLossMask, BertForSequenceClassification
# from src_paddle.pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

# from nn.data_parallel import DataParallelImbalance
from paddle import DataParallel as DataParallelImbalance
import src_paddle.biunilm.seq2seq_loader as seq2seq_loader
import paddle.distributed as dist
import paddle.distributed.fleet as fleet

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()

def _get_max_epoch_model(output_dir):
    fn_model_list = glob.glob(os.path.join(output_dir, "model.*.pdparams"))
    fn_optim_list = glob.glob(os.path.join(output_dir, "optim.*.pdparams"))
    if (not fn_model_list) or (not fn_optim_list):
        return None
    both_set = set([int(Path(fn).stem.split('.')[-1]) for fn in fn_model_list]
                   ) & set([int(Path(fn).stem.split('.')[-1]) for fn in fn_optim_list])
    if both_set:
        return max(both_set)
    else:
        return None

# Required parameters
parser.add_argument(
  "--recover_on",
  default=True,
  type = bool,
  help = "Decide whether Recover from checkpoint or not."
)
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
parser.add_argument("--vocab", default=None, type=str, required=True,
                    help="Pretrained vacab")
parser.add_argument("--config_path", default=None, type=str,
                    help="Bert config file path.")
parser.add_argument("--output_dir",
                    default=None,
                    type=str,
                    required=True,
                    help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--log_dir",
                    default='',
                    type=str,
                    required=True,
                    help="The output directory where the log will be written.")
parser.add_argument("--model_recover_path",
                    default=None,
                    type=str,
                    required=True,
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
                    default=30,
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

assert Path(args.model_recover_path).is_file(
    ), "--model_recover_path doesn't exist"

print(Path(args.model_recover_path))
assert Path(args.model_recover_path).is_file(
), "--model_recover_path doesn't exist"

args.output_dir = args.output_dir.replace(
    '[PT_OUTPUT_DIR]', os.getenv('PT_OUTPUT_DIR', ''))
args.log_dir = args.log_dir.replace(
    '[PT_OUTPUT_DIR]', os.getenv('PT_OUTPUT_DIR', ''))

os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(args.log_dir, exist_ok=True)
json.dump(args.__dict__, open(os.path.join(
    args.output_dir, 'opt.json'), 'w'), sort_keys=True, indent=2)
glue_tasks = ["qnli"]

if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

args.train_batch_size = int(
    args.train_batch_size / args.gradient_accumulation_steps)

glue_tasks_num_labels = {
    "qnli": 2
}

glue_task_type = {
    "qnli": "classification",
}

def load_glue_sub_data(name):
    if name not in glue_tasks:
        raise Exception("Name Error: name must in ", glue_tasks)
    
    splits = ("train", "dev", "test")
    
    if name == "mnli":
        splits = ("train", "dev_matched")
    
    dataset = load_dataset("glue", name=name, splits=splits)

    return dataset


task_name = "qnli"
train, dev, test = load_glue_sub_data(task_name)
print(train[:2])
print(test[:2])
print(dev[:2])

random.seed(42)
np.random.seed(42)
paddle.seed(42)
paddle.set_device("gpu:0")

if not args.do_train and not args.do_eval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True.")

#tokenizer = BertTokenizer.from_pretrained(
#        args.vocab, do_lower_case=args.do_lower_case)
tokenizer = paddlenlp.transformers.BertTokenizer.from_pretrained("bert-large-cased")

data_tokenizer = WhitespaceTokenizer() if args.tokenized_input else tokenizer


trans_func = partial(convert_example, task_name=task_name, tokenizer=tokenizer, max_seq_length=args.max_seq_length)
batchify_fn =  lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=0), # input
    Pad(axis=0, pad_val=0), # segment
    Stack(dtype="int64") if glue_task_type[task_name]=="classification" else Stack(dtype="float32") # label
): [data for data in fn(samples)]

train_data_loader = create_dataloader(train, mode="train", batchify_fn=batchify_fn, trans_fn=trans_func)
dev_data_loader = create_dataloader(dev, mode="dev", batchify_fn=batchify_fn, trans_fn=trans_func)

_state_dict = {} if args.from_scratch else None

cls_num_labels = 2
# type_vocab_size = 6 + \
#     (1 if args.s2s_add_segment else 0) if args.new_segment_ids else 2
type_vocab_size = 2

relax_projection = 4 if args.relax_projection else 0
num_sentlvl_labels = 2 if args.has_sentence_oracle else 0
# model = BertForPreTrainingLossMask.from_pretrained(
#             args.bert_model, state_dict=_state_dict, num_labels=cls_num_labels, num_rel=0, type_vocab_size=type_vocab_size, config_path=args.config_path, task_idx=3, num_sentlvl_labels=num_sentlvl_labels, max_position_embeddings=args.max_position_embeddings, label_smoothing=args.label_smoothing, fp32_embedding=args.fp32_embedding, relax_projection=relax_projection, new_pos_ids=args.new_pos_ids, ffn_type=args.ffn_type, hidden_dropout_prob=args.hidden_dropout_prob, attention_probs_dropout_prob=args.attention_probs_dropout_prob, num_qkv=args.num_qkv, seg_emb=args.seg_emb)


# Recover Model
recover_step = _get_max_epoch_model(args.output_dir)
t_total = int(len(train_data_loader) * args.num_train_epochs /
                  args.gradient_accumulation_steps)
if args.recover_on:
    if recover_step:
        logger.info("***** Recover model: %d *****", recover_step)
        model_recover = paddle.load(os.path.join(
            args.output_dir, "model.{0}.pdparams".format(recover_step)))
        # recover_step == number of epochs
        global_step = math.floor(
            recover_step * t_total / args.num_train_epochs)
    elif args.model_recover_path:
        logger.info("***** Recover model: %s *****",
                    args.model_recover_path)
        model_recover = paddle.load(
            args.model_recover_path)
        global_step = 0
else:
    model_recover = _state_dict


model = BertForSequenceClassification.from_pretrained(
    args.bert_model, state_dict=model_recover, 
    num_labels=cls_num_labels,
    type_vocab_size=type_vocab_size, config_path=args.config_path, 
    task_idx=3, 
    max_position_embeddings=args.max_position_embeddings, 
    fp32_embedding=args.fp32_embedding, 
    relax_projection=relax_projection, new_pos_ids=args.new_pos_ids, 
    ffn_type=args.ffn_type, hidden_dropout_prob=args.hidden_dropout_prob, 
    attention_probs_dropout_prob=args.attention_probs_dropout_prob, num_qkv=args.num_qkv   
)

# 设置lr_scheduler
num_training_steps = len(train_data_loader) * args.num_train_epochs
lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps, args.warmup_proportion)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
g_clip = paddle.nn.ClipGradByGlobalNorm(1.0)

optimizer = AdamW(
    learning_rate = lr_scheduler,
    parameters = optimizer_grouped_parameters,
    # weight_decay = 0.01,
    epsilon = 1e-6,
    grad_clip = g_clip
)
# recover optimizer
if args.recover_on:
    if recover_step:
        logger.info("***** Recover optimizer: %d *****", recover_step)
        optim_recover = paddle.load(os.path.join(
            args.output_dir, "optim.{0}.pdparams".format(recover_step)))
        if hasattr(optim_recover, 'state_dict'):
            optim_recover = optim_recover.state_dict()
        optimizer.set_state_dict(optim_recover)
        if args.loss_scale == 0:
            logger.info("***** Recover optimizer: dynamic_loss_scale *****")
            optimizer.dynamic_loss_scale = True

import paddle.nn.functional as F
from utils.metrics import compute_metrics

@paddle.no_grad()
def evaluate(model, data_loader, task_type="classification"):
    model.eval()
    losses = []
    preds = None
    out_labels = None
    for batch in data_loader:
        input_ids, token_type_ids, labels = batch
        logits = model(input_ids, token_type_ids)
        if task_type == "classification":
            loss = F.cross_entropy(logits, labels)
            losses.append(loss.numpy())

            if preds is None:
                preds = np.argmax(logits.detach().numpy(), axis=1).reshape([len(logits), 1])
                out_labels = labels.detach().numpy()
            else:
                preds = np.append(preds, np.argmax(logits.detach().numpy(), axis=1).reshape([len(logits), 1]), axis=0)
                out_labels = np.append(out_labels, labels.detach().numpy(), axis=0)
        else:
            loss = F.mse_loss(logits, labels)
            losses.append(loss.numpy())

            if preds is None:
                preds = logits.detach().numpy()
                out_labels = labels.detach().numpy()
            else:
                preds = np.append(preds, logits.detach().numpy(), axis=0)
                out_labels = np.append(out_labels, labels.detach().numpy(), axis=0)
    
    result = compute_metrics(task_name, preds.reshape(-1), out_labels.reshape(-1))
    print("evaluate result: ",result)

    model.train()
n_gpu = 1

def do_train():
    model.train()
    for  epoch in range(1, int(args.num_train_epochs)+1):
        for step, batch in enumerate(train_data_loader, start=1):
            input_ids, segment_ids, labels = batch
            logits = model(input_ids, segment_ids)
            # print("input_ids",input_ids)
            # print("segment_ids",segment_ids)
            # print("labels",labels)
            # print("loss_tuple",loss_tuple)
            # exit(0)
           
           
            if glue_task_type[task_name] == "classification":
                loss = F.cross_entropy(logits, labels)
            else:
                loss = F.mse_loss(logits, labels)

            if step%20 == 0:
                print("epoch: {}/{}, step: {}/{}, loss: {} ".format(epoch, int(args.num_train_epochs), step, len(train_data_loader), loss.numpy()))
                
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()
        
        logger.info("** ** * Saving fine-tuned model and optimizer ** ** * ")
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, "model.{0}.pdparams".format(epoch))
        paddle.save(model_to_save.state_dict(), output_model_file)
        output_optim_file = os.path.join( args.output_dir, "optim.{0}.pdparams".format(epoch))
        paddle.save(optimizer.state_dict(), output_optim_file)

        logger.info("***** CUDA.empty_cache() *****")
        # torch.cuda.empty_cache()

        evaluate(model, dev_data_loader, task_type=glue_task_type[task_name])



# 开始模型训练
do_train()
# evaluate(model, dev_data_loader, task_type=glue_task_type[task_name])

