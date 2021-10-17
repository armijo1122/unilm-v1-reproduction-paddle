import sys
import os
sys.path.append(os.path.dirname('/lustre/S/fuqiang/unilm/unilm/unilm-v1/src_paddle/'))
import torch
from collections import OrderedDict
import numpy as np
from pytorch_pretrained_bert.modeling import BertForPreTrainingLossMask,BertConfig


def export_weight_names(net):
    print(net.state_dict().keys())
    with open('paddle.txt', 'w') as f:
        for key in net.state_dict().keys():
            f.write(key + '\n')
            
            
bert_model = "/lustre/S/fuqiang/unilm/unilm/unilm-v1/bert-cased-pretrained-cache/bert-large-cased.tar.gz"
model_recover = torch.load(
               "/lustre/S/fuqiang/unilm/unilm/unilm-v1/bert-cased-pretrained-cache/unilm1-large-cased.bin", map_location='cpu')

config = BertConfig.from_json_file("/lustre/S/fuqiang/unilm/unilm/unilm-v1/bert-cased-pretrained-cache/bert_config.json")
#model = BertForPreTrainingLossMask.from_pretrained(bert_model, state_dict=model_recover, num_labels=2, num_rel=0, type_vocab_size=8, config_path=None, task_idx=3, num_sentlvl_labels=0, max_position_embeddings=192, label_smoothing=0.1, fp32_embedding=None, relax_projection=0, new_pos_ids=None, ffn_type=None, hidden_dropout_prob=None, attention_probs_dropout_prob=None, num_qkv=None, seg_emb=None)
model = BertForPreTrainingLossMask(config)
export_weight_names(model)
paddle_list = open('paddle.txt')
state_dict = torch.load('unilm1-large-cased.bin')

paddle_state_dict = OrderedDict()
paddle_list = paddle_list.readlines()
torch_list = state_dict.keys()
pa = "intermediate.dense.weight"
pb = "output.dense.weight"

with open('torch.txt','w') as f:
    for key in torch_list:
        f.write(key + '\n')      
for p in paddle_list:
    p = p.strip()
    t = p
    if "mean" in p:
        t = p.replace("_mean", "running_mean")
    if "variance" in p:
        t = p.replace("_variance", "running_var")
    if t in torch_list:
        if 'cls.predictions.decoder.weight' in p or 'cls.seq_relationship.weight' in p or 'self.query.weight' in p or 'self.key.weight' in p or 'self.value.weight' in p or 'pooler.dense.weight' in p:
            paddle_state_dict[p] = state_dict[t].detach().cpu().numpy().T.astype(np.float32)
        elif 'fc' not in p:
            paddle_state_dict[p] = state_dict[t].detach().cpu().numpy().astype(np.float32)
            if pa in p or pb in p:
                paddle_state_dict[p] = state_dict[t].detach().cpu().numpy().T.astype(np.float32)
        else:
            paddle_state_dict[p] = state_dict[t].detach().cpu().numpy().T.astype(np.float32)
    else:
        print(p)

f = open('unilm1-large-cased.pdparams', 'wb')
import pickle
pickle.dump(paddle_state_dict, f)
f.close()