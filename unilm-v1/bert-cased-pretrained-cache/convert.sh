export CUDA_VISIBLE_DEVICES=0,1,2,3
export NGPU=4;python -m torch.distributed.launch --nproc_per_node=$NGPU convert_torch_to_paddle.py