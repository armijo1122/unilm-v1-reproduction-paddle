import sys
import os
sys.path.append(os.path.dirname('/lustre/S/fuqiang/unilm/unilm/unilm-v1/src/'))
import torch
from collections import OrderedDict
import numpy as np
from pytorch_pretrained_bert.modeling import BertForPreTrainingLossMask
from reprod_log import ReprodLogger
            
            
bert_model = "/lustre/S/fuqiang/unilm/unilm/unilm-v1/bert-cased-pretrained-cache/bert-large-cased.tar.gz"
model_recover = torch.load(
               "/lustre/S/fuqiang/unilm/unilm/unilm-v1/bert-cased-pretrained-cache/unilm1-large-cased.bin", map_location='cpu')

model = BertForPreTrainingLossMask.from_pretrained(bert_model, state_dict=model_recover, num_labels=2, num_rel=0, type_vocab_size=8, config_path=None, task_idx=3, num_sentlvl_labels=0, max_position_embeddings=192, label_smoothing=0, fp32_embedding=None, relax_projection=0, new_pos_ids=None, ffn_type=None, hidden_dropout_prob=None, attention_probs_dropout_prob=None, num_qkv=None, seg_emb=None)

model.eval()
model.float()

random_input = torch.LongTensor([[5, 6, 1], [3, 5, 0]])
token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])

# forward
output1, output2 = model(random_input, token_type_ids, input_mask)
#print("Model Output:",output)
reprod_log_1 = ReprodLogger()
reprod_log_1.add("forward_output", output1.cpu().detach().numpy())
reprod_log_1.save("torch_forward.npy")

# loss
# lm_label_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
# masked_pos = torch.LongTensor([[0, 0, 1], [1, 1, 0]])
# masked_weights = torch.FloatTensor([[0.1,0.24,0.6], [0.3, 0.51, 0.8]])
# masked_lm_loss, next_sentence_loss = model(random_input, token_type_ids, input_mask, lm_label_ids, masked_pos=masked_pos, masked_weights = masked_weights)
# reprod_log_2 = ReprodLogger()
# reprod_log_2.add("masked_lm_loss", masked_lm_loss.cpu().detach().numpy())
# reprod_log_2.add("next_sentence_loss", np.array(next_sentence_loss))
# reprod_log_2.save("loss_torch.npy")