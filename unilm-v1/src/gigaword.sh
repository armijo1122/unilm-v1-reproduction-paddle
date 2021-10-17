# run fine-tuning
DATA_DIR='/lustre/S/fuqiang/unilm/unilm/unilm-v1/data/gigaword'
OUTPUT_DIR='/lustre/S/fuqiang/unilm/unilm/unilm-v1/giga_finetune_out/'
MODEL_RECOVER_PATH='/lustre/S/fuqiang/unilm/unilm/unilm-v1/bert-cased-pretrained-cache/unilm1-large-cased.bin'
export PYTORCH_PRETRAINED_BERT_CACHE='/lustre/S/fuqiang/unilm/unilm/unilm-v1/bert-cased-pretrained-cache'
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NGPU=4;python -m torch.distributed.launch --nproc_per_node=$NGPU biunilm/run_seq2seq.py --do_train --fp16 --amp --num_workers 0 --bert_model 'bert-large-cased' --new_segment_ids --tokenized_input --data_dir '/lustre/S/fuqiang/unilm/unilm/unilm-v1/data/gigaword' --src_file 'train.src.10k' --tgt_file 'train.tgt.10k' --output_dir '/lustre/S/fuqiang/unilm/unilm/unilm-v1/giga_finetune_out/bert_save' --log_dir '/lustre/S/fuqiang/unilm/unilm/unilm-v1/giga_finetune_out/bert_log' --model_recover_path '/lustre/S/fuqiang/unilm/unilm/unilm-v1/bert-cased-pretrained-cache/unilm1-large-cased.bin' --max_seq_length 192 --max_position_embeddings 192 --trunc_seg a --always_truncate_tail --max_len_b 64 --mask_prob 0.7 --max_pred 64 --train_batch_size 128 --gradient_accumulation_steps 1 --learning_rate 0.00001 --warmup_proportion 0.1 --label_smoothing 0.1 --num_train_epochs 30
