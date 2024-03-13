# Transformer_from_zero2one
Simple single-card training of transformer for machine translation <br>
This code is training with 4090 Nvidia GPU. If the version of your GPU is lower, please decrease the batch size.

## Installation 
pip install transformers <br>
pip install sentencepiece <br>

The dataset of *Helsinki-NLP* and *translation2019zh* is shared in the alipan. The link is shown below.<br>
<https://www.alipan.com/s/F5dRWAwod7w>

## Usage
Finetune huggingface model <br>
'python train_huggingface.py'

Test huggingface model <br>
'python test_huggingface.py'

Train from scratch <br>
'python train.py' <br>
We provide a trained model for you to test the performance. Download the model_weights.pt from the provided link, then create a folder in root folder called record.<br>
Finally, put the model_weights.pt into the record folder. Comment the train_loop() function, and uncomment the translate() function.