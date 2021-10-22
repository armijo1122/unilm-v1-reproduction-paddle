# Paddle复现论文"Unified Language Model Pre-training for Natural Language Understanding and Generation"

## 1. 文件组织方式说明

"src/"文件夹下存放的是原文pytorch代码，详见：https://github.com/microsoft/unilm

"src_paddle/"文件夹下存放的是复现后的Paddle代码，该文件夹下的"check_log"文件夹下存储了几个检查点的log文件，"torch_to_paddle_api/"文件夹下实现了个别pytorch api的Paddle重新实现

"bert-cased-pretrained-cache/"文件夹下是预训练模型转换的代码，其中convert_pretrained_to_paddle.py是将预训练模型pytorch_model.bin转换成Paddle模型，convert_torch_to_paddle.py是将unilm1-large-cased.bin转换成对应的Paddle权重文件，转换后得到的pyorch_model.pdparams和unilm1-large-cased.bin下载链接为：

"data_align/"文件夹下存放的是验证与测试数据集对齐时使用的文件

## 2. 测试

前向模型对齐：将下载好的权重文件放到"bert-cased-pretrained-cache"目录下，依次运行forwardt.sh，forwardp.sh分别进行PyTorch和Paddle的前向传播，然后运行check_forward.py检查对齐误差。

验证集，测试集对齐：读取数据集中的test.src和test.tgt文件，PyTorch读取方式详见data_align/valid_test_align.py，Paddle读取方式详见data_align/valid_test_align.py。由于数据集构建时采用的__getitem__方法采用了random.choice()取数据，因此最后运行data_align/check_test_align.py时显示没有对齐，但可以正常取数据。

损失函数对齐：测试文件与前向传播时相同，只是输入有所变化，仍然是执行forwardt.sh和forwardp.sh,只是此时将forward部分代码注释掉了。检查时执行check_forward.py即可。

优化器对齐：详见src_paddle/pytorch_pretrained_bert/optimization.py与optimization_fq16.py，目前还没来得及联合最后一步进行测试

