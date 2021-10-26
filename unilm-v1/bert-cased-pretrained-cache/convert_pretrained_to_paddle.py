import sys
import os
sys.path.append(os.path.dirname('/lustre/S/fuqiang/unilm/unilm/unilm-v1/src_paddle/'))
import torch
from collections import OrderedDict
import numpy as np
from pytorch_pretrained_bert.modeling import BertForPreTraining,BertConfig


def export_weight_names(net):
    print(net.state_dict().keys())
    with open('paddle.txt', 'w') as f:
        for key in net.state_dict().keys():
            f.write(key + '\n')
            
            
bert_model = "/lustre/S/fuqiang/unilm/unilm/unilm-v1/bert-cased-pretrained-cache/bert-large-cased.tar.gz"
model_recover = torch.load(
               "/lustre/S/fuqiang/unilm/unilm/unilm-v1/bert-cased-pretrained-cache/unilm1-large-cased.bin", map_location='cpu')

config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
model = BertForPreTraining(config)
export_weight_names(model)
paddle_list = open('paddle.txt')
state_dict = torch.load('pytorch_model.bin')
state_dict_new = {}

paddle_state_dict = OrderedDict()
paddle_list = paddle_list.readlines()
torch_list = state_dict.keys()
pa = "intermediate.dense.weight"
pb = "output.dense.weight"

for t in torch_list:
    state_dict_new[t] = state_dict[t]
    state_dict_new[t] = state_dict_new[t].detach().cpu().numpy().astype(np.float32)

for p in paddle_list:
    p = p.strip()
    t = p
    if "mean" in p:
        t = p.replace("_mean", "running_mean")
    if "variance" in p:
        t = p.replace("_variance", "running_var")
    if t in torch_list:
        if 'cls.predictions.decoder.weight' in p or 'cls.seq_relationship.weight' in p or 'self.query.weight' in p or 'self.key.weight' in p or 'self.value.weight' in p or 'pooler.dense.weight' in p or 'cls.predictions.transform.dense.weight' in p:
            paddle_state_dict[p] = state_dict[t].detach().cpu().numpy().T.astype(np.float32)
        elif 'fc' not in p:
            paddle_state_dict[p] = state_dict[t].detach().cpu().numpy().astype(np.float32)
            if pa in p or pb in p:
                paddle_state_dict[p] = state_dict[t].detach().cpu().numpy().T.astype(np.float32)
        else:
            paddle_state_dict[p] = state_dict[t].detach().cpu().numpy().T.astype(np.float32)
    else:
        print(p)

f = open('pytorch_model.pdparams', 'wb')
f_t = open('pytorch_model32.bin', 'wb')
import pickle
pickle.dump(paddle_state_dict, f)
pickle.dump(state_dict_new, f_t)
f.close()
f_t.close()