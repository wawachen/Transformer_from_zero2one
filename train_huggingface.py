from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from dataset import C2E_translate
from torch.utils.data import random_split, DataLoader
import torch
from sacrebleu.metrics import BLEU
from tqdm import tqdm
import numpy as np 
from tensorboardX import SummaryWriter
from time import localtime, strftime
import os
from transformers import AdamW, get_scheduler
import random 

def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(42)

bleu = BLEU()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

max_dataset_size = 220000
train_size = 200000
valid_size = 20000

path = "/home/wawa/pytorch-transformer/my_reproduce/translation2019zh/translation2019zh_train.json"
data = C2E_translate(path,limit_num=max_dataset_size)
train_data, valid_data = random_split(data,[train_size,valid_size])

path1 = "/home/wawa/pytorch-transformer/my_reproduce/translation2019zh/translation2019zh_valid.json"
test_data = C2E_translate(path1)

model_check_point = "Helsinki-NLP/opus-mt-zh-en"
tokenizer = AutoTokenizer.from_pretrained(model_check_point)

#four sentences
max_input_len = 128
max_target_len = 128

batch_size = 32
learning_rate = 1e-5
epoch_num = 3

def test_loop(dataloader, model):
    preds = []
    refs = []
    
    model.eval() 

    for batch_data in tqdm(dataloader):
        batch_data = batch_data.to(device)

        with torch.no_grad():
            pred_tokens = model.generate(batch_data['input_ids'], attention_mask = batch_data['attention_mask'], max_length = max_target_len).cpu().numpy()
        labels_tokens = batch_data['labels'].cpu().numpy()

        pred_s = tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)
        labels_tokens = np.where(labels_tokens!=-100, labels_tokens, tokenizer.pad_token_id)
        label_s = tokenizer.batch_decode(labels_tokens, skip_special_tokens=True)

        preds += [pre.strip() for pre in pred_s]
        refs+=[[lab.strip()] for lab in label_s]

    bleu_score = bleu.corpus_score(preds,refs).score
    print(f'The score is {bleu_score:>0.2f}\n')
    return bleu_score


def train_loop(dataloader,model,optimizer, lr_scheduler, epochs, logger, path):
    global_step = 0

    for epoch in range(epochs):
        model.train()
        batch_iterator = tqdm(dataloader,desc=f'Process Epoch : {epoch:02d}')
        for batch_data in batch_iterator:
            batch_data.to(device)
            pred = model(**batch_data)
            loss = pred.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            logger.add_scalar("train loss", loss.item(),global_step)

            global_step+=1

        score = test_loop(valid_dataloader,model)
        logger.add_scalar("score",score,epoch)

        save_dict = {"model_params":model.state_dict(),"optim_params":optimizer.state_dict(),'epoch':epoch, "global_step":global_step}
        save_path = f'{path}epoch_{epoch}_model_weights.pt'
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(save_dict, save_path)


# print(labels)
def shift_right(labels):
    return torch.cat([torch.ones(labels.shape[0],1,dtype=torch.int)*65000,labels[:,:-1]],dim=-1)

# manually process the batch data
def token_fn(samples):
    batch_inputs, batch_targets = [],[]
    for sample in samples:
        batch_inputs.append(sample['chinese'])
        batch_targets.append(sample['english'])

    batch_data = tokenizer(batch_inputs,padding=True,max_length=max_input_len,truncation=True, return_tensors="pt")
    with tokenizer.as_target_tokenizer():
        batch_data_t = tokenizer(batch_targets,padding=True,max_length=max_target_len, truncation=True, return_tensors="pt")
        labels = batch_data_t['input_ids']

    batch_data['decoder_input_ids'] = shift_right(labels)
    end_token_index = torch.where(labels==tokenizer.eos_token_id)[1] 
    for i,index in enumerate(end_token_index):
        labels[i, index+1:] = -100
    batch_data['labels'] = labels
    # print(decoder_mask.shape,causal_mask(decoder_mask.shape[0], decoder_mask.shape[1]).shape)
    #[batch,1,seq_len] x [batch, seq_len, seq_len]
    # batch_data['attention_mask'] = batch_data['attention_mask'].unsqueeze()
    # batch_data['wawa'] = decoder_mask
    # each batch data has "input_ids", "decoder_input_ids", "labels", "attention_mask"
    return batch_data

train_dataloader = DataLoader(train_data,batch_size = batch_size, shuffle=True, collate_fn=token_fn)
valid_dataloader = DataLoader(valid_data,batch_size = batch_size, shuffle=False, collate_fn=token_fn)

test_dataloader = DataLoader(test_data,batch_size=batch_size, shuffle=False,collate_fn=token_fn)

# batch = next(iter(train_dataloader))
# print('batch_shape: ', {k:v.shape for k,v in batch.items()})
# print(batch)

#call huggingface model
model = AutoModelForSeq2SeqLM.from_pretrained(model_check_point)
model.to(device)

path_log = os.path.join("/home/wawa/pytorch-transformer/my_reproduce", strftime("%Y-%m-%d--%H:%M:%S", localtime()))
path_model = path_log+"/model/"

if not os.path.exists(path_log):
    os.makedirs(path_log)

logger = SummaryWriter(path_log)

optimizer = AdamW(model.parameters(),lr=learning_rate)
lr_scheduler = get_scheduler("linear",optimizer=optimizer,num_warmup_steps=0, num_training_steps=len(train_dataloader)*epoch_num)

train_loop(train_dataloader, model, optimizer, lr_scheduler, epoch_num, logger, path_model)


logger.close()

# sentence = "我叫睿，我现在在谢菲尔德大学工程学院攻读博士学位。"
# sen_inputs = tokenizer(sentence,return_tensors='pt').to(device)
# sen_outputs = model.generate(sen_inputs['input_ids'],attention_mask=sen_inputs['attention_mask'], max_length = 128)

# sen_decoded = tokenizer.decode(sen_outputs[0],skip_special_tokens=True)
# print(sen_inputs['input_ids'].shape)
# test_loop(test_dataloader,model)




