import sys
import os
sys.path.append(os.path.dirname('/lustre/S/fuqiang/unilm/unilm/unilm-v1/src_paddle/'))
import paddle
from collections import OrderedDict
import numpy as np
from pytorch_pretrained_bert.modeling import BertForPreTrainingLossMask, BertForSequenceClassification
from reprod_log import ReprodLogger
from pytorch_pretrained_bert.optimization import BertAdam
from paddle.optimizer import Adamax,AdamW
            
            
bert_model = "/lustre/S/fuqiang/unilm/unilm/unilm-v1/bert-cased-pretrained-cache/bert-large-cased-paddle.tar.gz"
model_recover = paddle.load(
               "/lustre/S/fuqiang/unilm/unilm/unilm-v1/bert-cased-pretrained-cache/unilm1-large-cased.pdparams")

# model = BertForPreTrainingLossMask.from_pretrained(bert_model, state_dict=model_recover, num_labels=2, num_rel=0, type_vocab_size=8, config_path=None, task_idx=3, num_sentlvl_labels=0, max_position_embeddings=192, label_smoothing=0, fp32_embedding=None, relax_projection=0, new_pos_ids=None, ffn_type=None, hidden_dropout_prob=None, attention_probs_dropout_prob=None, num_qkv=None, seg_emb=None)
model = BertForSequenceClassification.from_pretrained(
    bert_model, state_dict=model_recover, 
    num_labels=2,
    type_vocab_size=8, config_path=None, 
    task_idx=3, 
    max_position_embeddings=192, 
    fp32_embedding=None, 
    relax_projection=0, new_pos_ids=None, 
    ffn_type=None, hidden_dropout_prob=None, 
    attention_probs_dropout_prob=None, num_qkv=None   
)
model.eval()


random_input = paddle.to_tensor([[5, 6, 1], [3, 5, 0]])
token_type_ids = paddle.to_tensor([[0, 0, 1], [0, 1, 0]])
input_mask = paddle.to_tensor([[1, 1, 1], [1, 1, 0]])

# forward
# output1, output2 = model(random_input, token_type_ids, input_mask)
#print("Model Output:",output)
# reprod_log_1 = ReprodLogger()
# reprod_log_1.add("forward_output", output1.cpu().detach().numpy())
# reprod_log_1.save("paddle_forward.npy")

# loss
lm_label_ids = paddle.to_tensor([[0, 0, 1], [0, 1, 0]])
masked_pos = paddle.to_tensor([[0, 0, 1], [1, 1, 0]])
masked_weights = paddle.to_tensor([[0.1, 0.24, 0.6], [0.3, 0.51, 0.8]], stop_gradient = False)
model.train()

param_optimizer = list(model.named_parameters())
# for n, p in param_optimizer:
#     print("KEYS:", n)
#     print("VALUES:", p.shape)
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(
        nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(
        nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

# for n,p in model.named_parameters():
#     print(n, p.shape)
# exit(0)
decay_params = [
    p.name
    for n, p in model.named_parameters()
    if not any(nd in n for nd in ['bias', 'LayerNorm.bias', 'LayerNorm.weight'])
]

g_clip = paddle.nn.ClipGradByGlobalNorm(1.0)

optimizer = AdamW(
    learning_rate = 0.00001,
    parameters = optimizer_grouped_parameters,
    # weight_decay = 0.01,
    epsilon = 1e-6
    # grad_clip = g_clip
)

reprod_log_3 = ReprodLogger()
import paddle.nn.functional as F
glue_task_type = {
    "qnli": "classification",
}
task_name = "qnli"
labels = paddle.to_tensor([0,1]).astype('int64')
for idx in range(100):
    
    # with paddle.no_grad():
    logits = model(random_input, token_type_ids, input_mask)
    
    if glue_task_type[task_name] == "classification":
        loss = F.cross_entropy(logits, labels)
    else:
        loss = F.mse_loss(logits, labels)


    # optimizer = BertAdam(
    #     optimizer_grouped_parameters,
    #     weight_decay = 0.01,
    #     learning_rate = 0.00001
    # )
    # parameters = []

    # for group in optimizer_grouped_parameters:
    #     # if(group.keys()!='params'):
    #     #     print("GROUP DICT:", group.items())
    #     print("GROUP DICT:", group.keys())
    #     for p in group['params']:
    #         parameters.append(p)
    #     weight_decay = group["weight_decay"]
    #     print("weight_decay:", weight_decay)
    # parameters = list()
    
    # reprod_log_2 = ReprodLogger()
    # reprod_log_2.add("masked_lm_loss", masked_lm_loss.cpu().detach().numpy())
    # reprod_log_2.add("next_sentence_loss", np.array(next_sentence_loss))
    # reprod_log_2.save("loss_paddle.npy")
    
    # loss = masked_lm_loss + next_sentence_loss
    
    # print("LOSS:", masked_lm_loss)
    # print("LOSS-PART:", loss)
    loss.backward()
    print("LOSS.GRAD:", loss.grad)
    print("LOSS:", loss)
    optimizer.step()
    optimizer.clear_grad()
    # print("!!!BACK:", back)
    
    # reprod_log_3.add("backward-LOSS-PART",masked_lm_loss.cpu().detach().numpy())
    lr = optimizer.get_lr()
    reprod_log_3.add("backward-LR{idx}".format(idx = idx), np.array(lr))
    reprod_log_3.add("backward-LOSS{idx}".format(idx = idx),loss.cpu().detach().numpy())
reprod_log_3.save("back_paddle.npy")