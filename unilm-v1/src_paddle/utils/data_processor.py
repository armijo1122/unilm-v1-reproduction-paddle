import paddle
import numpy as np

glue_task_type = {
    "qnli": "classification",
}
def convert_example( example, tokenizer, task_name=None, max_seq_length=512, is_test=False):
    if task_name in ["cola", "sst-2"]:
        encoded_inputs = tokenizer(text=example["sentence"], max_seq_len=max_seq_length)
        input_ids = encoded_inputs["input_ids"]
        token_type_ids = encoded_inputs["token_type_ids"]
        label = np.array([example["labels"]], dtype="int64")
        
    else:
        encoded_inputs = tokenizer(text=example["sentence1"], text_pair=example["sentence2"], max_seq_len=max_seq_length)
        input_ids = encoded_inputs["input_ids"]
        token_type_ids = encoded_inputs["token_type_ids"]
        if glue_task_type[task_name] == "classification":
            label = np.array([example["labels"]], dtype="int64")
        else:
            label = np.array([example["labels"]], dtype="float32")

    return input_ids, token_type_ids, label


def create_dataloader(dataset,
                      mode='train',
                      batch_size= 32,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)

