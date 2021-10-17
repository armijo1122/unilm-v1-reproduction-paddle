import torch
import sys
import os
sys.path.append(os.path.dirname('/lustre/S/fuqiang/unilm/unilm/unilm-v1/src/'))
torch_model = AlexNet()
torch_state_dict = load_state_dict_from_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
torch_model.load_state_dict(torch_state_dict)
torch_model.eval()