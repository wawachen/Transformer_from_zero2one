# Transformer_from_zero2one
Simple single-card training of transformer for machine translation <br>
This code is training with 4090 Nvidia GPU. If the version of your GPU is lower, please decrease the batch size.

## Installation 
pip install transformers <br>
pip install sentencepiece <br>
pip install sacrebleu 

The dataset of *Helsinki-NLP* and *translation2019zh* is shared in the alipan. The link is shown below.<br>
<https://www.alipan.com/s/F5dRWAwod7w>

## Usage
Finetune huggingface model <br>
> python train_huggingface.py

Test huggingface model <br>
> python test_huggingface.py

Train from scratch <br>
> python train.py <br>

We provide a trained model for you to test the performance of our scratched model. Download the model_weights.pt from the provided link below, <br>
<https://pan.baidu.com/s/1lGRzWyrK0_IxgDPTqVPgcQ> 提取码: tran 

Then create a folder in root folder called *record*.<br>
Finally, put the *model_weights.pt* into the *record* folder. 

To train the model, set **train_flag = 1**. To test the model, set set **train_flag = 0**