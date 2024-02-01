from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from dataset import C2E_translate
from torch.utils.data import DataLoader
import torch
from sacrebleu.metrics import BLEU
from tqdm import tqdm
import numpy as np 
import json

bleu = BLEU()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model_check_point = "Helsinki-NLP/opus-mt-zh-en"
tokenizer = AutoTokenizer.from_pretrained(model_check_point)

model = AutoModelForSeq2SeqLM.from_pretrained(model_check_point)
loaded_path = "/home/wawa/pytorch-transformer/my_reproduce/2024-01-12--20:55:19/model/epoch_2_model_weights.pt"
model.load_state_dict(torch.load(loaded_path)['model_params'])
model.to(device)

#four sentences
max_input_len = 128
max_target_len = 128

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

    return batch_data

path1 = "/home/wawa/pytorch-transformer/my_reproduce/translation2019zh/translation2019zh_valid.json"
test_data = C2E_translate(path1)
test_dataloader = DataLoader(test_data,batch_size = 32, shuffle= False, collate_fn=token_fn)

preds = []
refs = []
source = []

model.eval() 

for batch_data in tqdm(test_dataloader):
    batch_data = batch_data.to(device)

    with torch.no_grad():
        pred_tokens = model.generate(batch_data['input_ids'], attention_mask = batch_data['attention_mask'], max_length = max_target_len).cpu().numpy()
    labels_tokens = batch_data['labels'].cpu().numpy()

    src = tokenizer.batch_decode(batch_data['input_ids'],skip_special_tokens=True)
    pred_s = tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)
    labels_tokens = np.where(labels_tokens!=-100, labels_tokens, tokenizer.pad_token_id)
    label_s = tokenizer.batch_decode(labels_tokens, skip_special_tokens=True)

    source += [inp.strip() for inp in src]
    preds += [pre.strip() for pre in pred_s]
    refs+=[[lab.strip()] for lab in label_s]

bleu_score = bleu.corpus_score(preds,refs).score
print(f'The score is {bleu_score:>0.2f}\n')

result = {"source_sentence": source[0], "pred_sentence": preds[0], "ground_truth": refs[0][0]}

with open(f'{loaded_path}results.json','wt',encoding='utf-8') as f:
    f.write(json.dump(result,ensure_ascii=False))


