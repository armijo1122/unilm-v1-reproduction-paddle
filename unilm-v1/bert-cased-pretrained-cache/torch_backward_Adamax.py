import sys
import os
sys.path.append(os.path.dirname('/lustre/S/fuqiang/unilm/unilm/unilm-v1/src/'))
import torch
from collections import OrderedDict
import numpy as np
from pytorch_pretrained_bert.modeling import BertForPreTrainingLossMask
from reprod_log import ReprodLogger
from pytorch_pretrained_bert.optimization import BertAdam
            
            
bert_model = "/lustre/S/fuqiang/unilm/unilm/unilm-v1/bert-cased-pretrained-cache/bert-large-cased.tar.gz"
model_recover = torch.load(
               "/lustre/S/fuqiang/unilm/unilm/unilm-v1/bert-cased-pretrained-cache/unilm1-large-cased.bin", map_location='cpu')

model = BertForPreTrainingLossMask.from_pretrained(bert_model, state_dict=model_recover, num_labels=2, num_rel=0, type_vocab_size=8, config_path=None, task_idx=3, num_sentlvl_labels=0, max_position_embeddings=192, label_smoothing=0, fp32_embedding=None, relax_projection=0, new_pos_ids=None, ffn_type=None, hidden_dropout_prob=None, attention_probs_dropout_prob=None, num_qkv=None, seg_emb=None)

model.eval()
model.float()

random_input = torch.tensor([[5, 6, 1], [3, 5, 0]], dtype = torch.long)
token_type_ids = torch.tensor([[0, 0, 1], [0, 1, 0]], dtype = torch.long)
input_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype = torch.long)

# forward
# output1, output2 = model(random_input, token_type_ids, input_mask)

#print("Model Output:",output)
# reprod_log_1 = ReprodLogger()
# reprod_log_1.add("forward_output", output1.cpu().detach().numpy())
# reprod_log_1.save("torch_forward.npy")

# loss
lm_label_ids = torch.tensor([[0, 0, 1], [0, 1, 0]], dtype = torch.long)
masked_pos = torch.tensor([[0, 0, 1], [1, 1, 0]], dtype = torch.long)
masked_weights = torch.tensor([[0.1,0.24,0.6], [0.3, 0.51, 0.8]], dtype = torch.float, requires_grad = True)
reprod_log_3 = ReprodLogger()
for idx in range(5):
    masked_lm_loss, next_sentence_loss = model(random_input, token_type_ids, input_mask, lm_label_ids, masked_pos=masked_pos, masked_weights = masked_weights)
    masked_lm_loss.retain_grad()
    # reprod_log_2 = ReprodLogger()
    # reprod_log_2.add("masked_lm_loss", masked_lm_loss.cpu().detach().numpy())
    # reprod_log_2.add("next_sentence_loss", np.array(next_sentence_loss))
    # reprod_log_2.save("loss_torch.npy")
    
    param_optimizer = list(model.named_parameters())
    # for n, p in param_optimizer:
    #     print("KEYS:", n)
    #     print("VALUES:", p.size())
    
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # optimizer_grouped_parameters = [
    #     {'params': model.parameters(), 'weight_decay': 0.01},
    # ]
    # backward
    # optimizer = BertAdam(
    #     optimizer_grouped_parameters,
    #     # weight_decay = 0.0,
    #     lr = 0.00001
    # )
    optimizer = torch.optim.Adamax(
        params=optimizer_grouped_parameters,
        lr=0.00001,
        betas=(0.9, 0.999),
        eps = 1e-6
    )
    loss = masked_lm_loss + next_sentence_loss
    back = loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    # print("LOSS:", masked_lm_loss)
    print("LOSS-PART:", loss)
    print("LOSS.GRAD:", masked_lm_loss.grad)
    print("!!!BACK:", back)
    
    # reprod_log_3.add("backward-LOSS-PART", masked_lm_loss.cpu().detach().numpy())
    reprod_log_3.add("backward-LOSS{idx}".format(idx = idx), loss.cpu().detach().numpy())
reprod_log_3.save("back_torch.npy")