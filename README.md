# Paddle复现论文"Unified Language Model Pre-training for Natural Language Understanding and Generation"

## 1. 文件组织方式说明

"src/"文件夹中存放原论文的PyTorch源代码，详见https://github.com/microsoft/unilm

"src_paddle/"文件夹中存放Paddle复现原文后的源代码，在该文件夹的子文件夹中，check_log文件夹下存放的是对齐测试的log文件，pytorch_pretrained_bert文件夹下存放的是Paddle实现的模型代码。torch_to_paddle_api文件夹下存放了部分PyTorch api的Paddle复现版本。

"data_align/"文件夹主要用于数据加载部分的对齐测试

"bert-cased-pretrained-cache/"文件夹用于前向对齐和损失函数对齐等步骤。

## 2. 转换后的模型文件和词汇表下载链接

链接：https://pan.baidu.com/s/143Lb12BS_36ztjXTywBZJg 

提取码：n0it

## 3. 对齐测试

### 模型权重文件转换：

bash convert.sh # 先运行convert_pretrained_to_paddle.py

bash convert.sh # 再运行convert_torch_to_paddle.py

说明：以上两步分别生成pytorch_model.pdparams和unilm1-large-cased-paddle.pdparams。原PyTorch模型和转换后的Paddle模型的参数名称列表分别保存在bert-cased-pretrained-cache文件夹下的torch.txt和paddle.txt文件中。

### 前向传播对齐：

cd bert-cased-pretrained-cache/

bash forwardt.sh # pytorch前向传播

bash forwardp.sh # paddle前向传播

python check_forward.py # 对齐精度检查

说明：由于原论文提供的权重文件数据类型为fp16，因此前向传播对齐误差大概在1e-4量级。

### 验证集和测试集对齐：

cd data_align/

bash align_test.sh # pytorch加载数据

bash align_test_p.sh # paddle加载数据

python check_test_align.py #对齐精度检查

说明：由于原文构建数据集时__getitem__方法采用了choice函数，每次取的数据随机，因此对齐检查时不通过，但复现后的dataset与loader可以正常加载数据。

### 损失函数和优化器对齐：

cd bert-cased-pretrained-cache/

bash forwardt.sh # pytorch前向传播计算损失

bash forwardp.sh # paddle前向传播计算损失

python check_forward.py # 对齐精度检查

说明：这一步操作采用的脚本与前向对齐相同，只是模型输入不同，优化器对齐的测试详见反向对齐部分，对齐精度详见。

### 反向对齐：

cd bert-cased-pretrained-cache/

python torch_backward.py # pytorch反向传播100轮

python paddle_backward.py # paddle反向传播100轮

python check_backward.py # 对齐精度检查

说明：反向对齐精度详见src_unilm-v1/check_log/check_backward.log

## 4. 训练

首先将模型权重文件下载到bert-cased-pretrained-cache/目录下，然后执行如下命令：

cd unilm-v1/

mkdir giga_finetune_out

cd giga_finetune_out/

mkdir bert_log/

cd src_paddle/

bash qnli.sh # 注意将脚本中涉及的路径更换成自己的目录

说明：训练的log文件详见src_paddle/check_log/unilm_train_log.log

训练后的模型文件下载地址：链接：https://pan.baidu.com/s/1yMBo9AjIsjVzM4GU6IOwIg 
提取码：b0uf

如果要运行评估模式可以直接在加载模型后运行bash qnli.sh，运行第一步将会给出评估分数。








