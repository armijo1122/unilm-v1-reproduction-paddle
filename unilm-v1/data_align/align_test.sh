export CUDA_VISIBLE_DEVICES=0
export NGPU=1;python -m torch.distributed.launch --nproc_per_node=$NGPU valid_test_align.py --data_dir "/lustre/S/fuqiang/unilm/unilm/unilm-v1/data/gigaword/" --src_file \
"test.src" --tgt_file "test.tgt" --bert_model "/lustre/S/fuqiang/unilm/unilm/unilm-v1/data_align/bert-large-cased-vocab/" --do_lower_case \
--max_position_embeddings 192 --tokenized_input --max_pred 64 --mask_prob 0.7 --new_segment_ids \
--trunc_seg a --always_truncate_tail --max_len_a 0 --max_len_b 64 \
--num_workers 0 --train_batch_size 1