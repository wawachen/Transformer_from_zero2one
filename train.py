from transformers import AutoTokenizer
from dataset import C2E_translate
from torch.utils.data import random_split, DataLoader
import torch
from model import build_transformer
from tqdm import tqdm
import numpy as np
from sacrebleu.metrics import BLEU

from tensorboardX import SummaryWriter
from time import localtime, strftime
import os
from transformers import AdamW, get_scheduler
import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

bleu = BLEU()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'using {device}')

max_dataset_size = 120000
train_size = 119990
valid_size = 10

path = "/home/wawa/pytorch-transformer/my_reproduce/translation2019zh/translation2019zh_train.json"
data = C2E_translate(path,limit_num=max_dataset_size)
train_data, valid_data = random_split(data,[train_size,valid_size])

path1 = "/home/wawa/pytorch-transformer/my_reproduce/translation2019zh/translation2019zh_valid.json"
test_data = C2E_translate(path1)

model_check_point = "Helsinki-NLP/opus-mt-zh-en"
tokenizer = AutoTokenizer.from_pretrained(model_check_point)


#single sentence
# input = tokenizer(train_data[0]['chinese'])
# print(train_data[0]['chinese'])
# print(input)
# print(tokenizer.convert_ids_to_tokens(input['input_ids']))

# with tokenizer.as_target_tokenizer():
#     target = tokenizer(train_data[0]['english']) 

# print(train_data[0]['english'])
# print(target)
# print(tokenizer.convert_ids_to_tokens(target['input_ids']))

#four sentences
max_input_len = 128
max_target_len = 128

batch_size = 16
learning_rate = 3e-4
epoch_num = 60

# inputs = [train_data[i]['chinese'] for i in range(5)]
# targets = [train_data[i]['english'] for i in range(5)]

# model_inputs = tokenizer(inputs,padding=True,max_length=max_input_len,truncation=True,return_tensors="pt")
# with tokenizer.as_target_tokenizer():
#     labels = tokenizer(targets,padding=True,max_length=max_target_len,truncation=True,return_tensors='pt')["input_ids"]
# print(tokenizer.bos_token_id)
# print(tokenizer.eos_token_id)
# print(tokenizer.pad_token_id)

# def greedy_decode(model, source, source_mask, max_length):
#     bos_id = 65000
#     eos_id = tokenizer.eos_token_id 
#     pad_id = tokenizer.pad_token_id

#     #(batch,seq,d_model)
#     with torch.no_grad():
#         encoder_output = model.encode(source, source_mask)

#     # (batch, max_length)
#     store_arr = torch.empty(source.shape[0],max_length).fill_(pad_id).type(source.type())
#     # (batch, 1)
#     decoder_input = torch.empty(source.shape[0],1).fill_(bos_id).type(source.type()).to(device)
#     l = 0
#     index = torch.arange(store_arr.shape[0]).int()

#     while decoder_input.shape[1] <= max_length:
#         #(batch,1,seq,seq)
#         decoder_mask = causal_mask(decoder_input.shape[0], decoder_input.shape[1]).unsqueeze(1).type(source.type()).to(device)
#         #(batch,seq,d_model)
#         decoder_output = model.decode(decoder_input, encoder_output[index,:,:], source_mask[index,:,:,:], decoder_mask)

#         proj_output = model.project(decoder_output[:,-1:,:]) #[batch, 1, vocab_size]

#         _, batch_word = torch.max(proj_output, dim=-1) #(batch, 1)
#         store_arr[index, l] = batch_word.squeeze(1)
#         # print(store_arr)
#         l+=1
#         index = np.where(batch_word.cpu().numpy()!=eos_id)[0]
#         decoder_input = torch.cat([decoder_input[index,:],batch_word[index,:]],dim=-1)
    
#     return store_arr

def beam_search(model, source, source_mask, ori_source_mask, max_length, beam_size):
    bos_id = 65000
    eos_id = tokenizer.eos_token_id 

    encoder_output = model.encode(source, source_mask)
    candidates = [(torch.empty(1,1).fill_(bos_id).type(source.type()).to(device), 1)]

    while True:
        if any([candidate.shape[1] == max_length for candidate,_ in candidates]):
            break
        
        new_candidates = []

        for candidate, score in candidates:
            if candidate[0,-1].item() == eos_id:
                continue

            decoder_attention_mask = torch.ones(1, candidate.shape[1]).type(source.type()).to(device)
            decoder_mask = generate_mask(decoder_attention_mask,decoder_attention_mask)
            cross_mask = generate_mask(decoder_attention_mask,ori_source_mask)
            decoder_output = model.decode(candidate, encoder_output, cross_mask, decoder_mask)

            proj_output = model.project(decoder_output[:,-1]) #(batch, vocab_size)
            probs, inds = torch.topk(proj_output, beam_size, dim = 1)
            
            for i in range(beam_size):
                new_candidates.append((torch.cat([candidate, inds[:,i].unsqueeze(0)],dim=-1), probs[:,i].item()+score))

        candidates = sorted(new_candidates,key=lambda x: x[1],reverse=True)
        candidates = candidates[:beam_size]

        if all([candidate[:,-1] == eos_id for candidate, _ in candidates]):
            break
    
    return candidates[0][0]
            
def greedy_decode(model, source, source_mask, ori_source_mask, max_length):
    bos_id = 65000
    eos_id = tokenizer.eos_token_id 

    encoder_output = model.encode(source, source_mask)

    decoder_input = torch.empty(1,1).fill_(bos_id).type(source.type()).to(device)

    while True:
        if decoder_input.shape[1] == max_length:
            break
        
        decoder_attention_mask = torch.ones(1,decoder_input.shape[1]).type(source.type()).to(device)
        decoder_mask = generate_mask(decoder_attention_mask,decoder_attention_mask)
        cross_mask = generate_mask(decoder_attention_mask,ori_source_mask)
        decoder_output = model.decode(decoder_input, encoder_output, cross_mask, decoder_mask)

        proj_output = model.project(decoder_output[:,-1])
        _, next_word = torch.max(proj_output, dim=-1,keepdim=True) #(batch, 1)
        decoder_input = torch.cat([decoder_input, next_word],dim=-1)

        if next_word == eos_id:
            break

    return decoder_input


def test_loop(dataloader, model):
    preds = []
    refs = []
    srcs = []
    
    model.eval() 

    with torch.no_grad():
        for batch_data in tqdm(dataloader):
            batch_data = batch_data.to(device)

            batch_encoder_input = batch_data['input_ids']
            batch_encoder_mask = batch_data['attention_mask']
            batch_labels = batch_data['labels'].cpu().numpy() # [batch,seq]
            batch_ori_encoder_mask = batch_data['original_attention_mask']

            assert batch_encoder_input.shape[0] == 1

            #[batch,max_seq]
            # pred_tokens = greedy_decode(model, batch_encoder_input, batch_encoder_mask, batch_ori_encoder_mask, max_target_len)
            pred_tokens = beam_search(model, batch_encoder_input, batch_encoder_mask, batch_ori_encoder_mask, max_target_len, 4)

            pred_tokens_list = [pred_tokens[i,:].cpu().numpy() for i in range(pred_tokens.shape[0])]

            source_tokens_list = [batch_encoder_input[i,:].cpu().numpy() for i in range(batch_encoder_input.shape[0])]
            source_sens = tokenizer.batch_decode(source_tokens_list, use_source_tokenizer=True, skip_special_tokens=True)

            pred_sens = tokenizer.batch_decode(pred_tokens_list, skip_special_tokens=True)
            labels_tokens = np.where(batch_labels!=-100, batch_labels, tokenizer.pad_token_id)
            labels_tokens_list = [labels_tokens[i,:] for i in range(labels_tokens.shape[0])]
            label_sens = tokenizer.batch_decode(labels_tokens_list, skip_special_tokens=True)
        

            preds += [pre.strip() for pre in pred_sens]
            refs+=[[lab.strip()] for lab in label_sens]
            srcs+=[src.strip() for src in source_sens]

            print(f'Source: {source_sens[0]}')
            print(f'Prediction: {pred_sens[0]}')
            print(f'Label: {label_sens[0]}')

    bleu_score = bleu.corpus_score(preds,refs).score
    print(f'The score is {bleu_score:>0.2f}\n')
    return bleu_score

def train_loop(dataloader, dataloader1, model,optimizer, epochs, logger, path):
    global_step = 0
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(epochs):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(dataloader,desc=f'Process Epoch : {epoch:02d}')
        for batch_data in batch_iterator:
            batch_encoder_input = batch_data['input_ids'].to(device)
            batch_encoder_mask = batch_data['attention_mask'].to(device)
            batch_decoder_input = batch_data['decoder_input_ids'].to(device)
            batch_decoder_mask = batch_data['decoder_mask'].to(device)
            batch_decoder_cross_mask = batch_data['decoder_cross_attention_mask'].to(device)
            batch_labels = batch_data['labels'].to(device) # [batch,seq]

            context = model.encode(batch_encoder_input,batch_encoder_mask)
            decoder_output = model.decode(batch_decoder_input,context,batch_decoder_cross_mask,batch_decoder_mask)
            proj_output = model.project(decoder_output) #[batch, seq, vocab_size]

            # print(proj_output.view(-1,tokenizer.vocab_size).shape, batch_labels.view(-1).shape)
            loss = loss_fn(proj_output.view(-1,tokenizer.vocab_size),batch_labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            # lr_scheduler.step()

            logger.add_scalar("train loss", loss.item(),global_step)

            global_step+=1

        score = test_loop(dataloader1, model)
        logger.add_scalar("score",score,epoch)

        if epoch != 0 and epoch%10== 0:
            save_dict = {"model_params":model.state_dict(),"optim_params":optimizer.state_dict(),'epoch':epoch, "global_step":global_step}
            save_path = f'{path}epoch_{epoch}_model_weights.pt'
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(save_dict, save_path)


# print(labels)
def shift_right(labels):
    return torch.cat([torch.ones(labels.shape[0],1,dtype=torch.int)*65000,labels[:,:-1]],dim=-1)

##################################################
def generate_mask(q_mask, k_mask, if_decoder = False):
    # q_pad shape: [n, q_len]
    # k_pad shape: [n, k_len]
    # 0 is the padding 
    # q_pad k_pad dtype: bool
    # pad_id = tokenizer.pad_token_id
  
    n, q_len = q_mask.shape
    n, k_len = k_mask.shape

    mask_shape = (n, 1, q_len, k_len)

    q_pad = q_mask == 0 #[n, q_len]
    k_pad = k_mask == 0 #[n, k_len]

    if if_decoder:
        mask = torch.tril(torch.ones(mask_shape))
    else:
        mask = torch.ones(mask_shape)
    
    mask = mask.to(q_pad.device)

    for i in range(n):
        mask[i, :, q_pad[i], :] = 0
        mask[i, :, :, k_pad[i]] = 0

    mask = mask.to(torch.bool)
    return mask

# manually process the batch data
def token_fn(samples):
    batch_inputs, batch_targets = [],[]
    for sample in samples:
        batch_inputs.append(sample['chinese'])
        batch_targets.append(sample['english'])

    batch_data = tokenizer(batch_inputs,padding=True,max_length=max_input_len,truncation=True, return_tensors="pt")
    encoder_mask = batch_data['attention_mask']
    batch_data['original_attention_mask'] = encoder_mask
    with tokenizer.as_target_tokenizer():
        batch_data_t = tokenizer(batch_targets,padding=True,max_length=max_target_len, truncation=True, return_tensors="pt")
        labels = batch_data_t['input_ids']
        decoder_mask = batch_data_t['attention_mask']
        # batch_data['target_attention_mask'] = batch_data_t['attention_mask'].unsqueeze(1).unsqueeze(1)

    batch_data['decoder_input_ids'] = shift_right(labels)
    end_token_index = torch.where(labels==tokenizer.eos_token_id)[1] 
    for i,index in enumerate(end_token_index):
        labels[i, index+1:] = -100
    batch_data['labels'] = labels
    # print(decoder_mask.shape,causal_mask(decoder_mask.shape[0], decoder_mask.shape[1]).shape)
    #[batch,1,1,seq_len] & [batch, 1, seq_len, seq_len]-> [batch, 1, seq_len, seq_len]
    batch_data['decoder_mask'] = generate_mask(decoder_mask,decoder_mask,if_decoder=True)
    batch_data['attention_mask'] = generate_mask(encoder_mask,encoder_mask) #[batch,1,1,seq_len]
    batch_data['decoder_cross_attention_mask'] = generate_mask(decoder_mask,encoder_mask)
    # batch_data['wawa'] = decoder_mask
    # each batch data has "input_ids", "decoder_input_ids", "labels", "attention_mask"
    return batch_data

train_dataloader = DataLoader(train_data,batch_size = batch_size, shuffle=True, collate_fn=token_fn)
valid_dataloader = DataLoader(valid_data, batch_size = 1, shuffle=True, collate_fn=token_fn)
test_dataloader = DataLoader(test_data,batch_size=2, shuffle=False,collate_fn=token_fn)

# batch = next(iter(test_dataloader))

# torch.set_printoptions(profile="full")
# np.set_printoptions(threshold=np.inf)

# print('batch_shape: ', {k:v.shape for k,v in batch.items()})
# print(batch)

#Configurations of transformer
d_model = 512
seq_len_encoder = max_input_len
seq_len_decoder = max_target_len
vocab_size_encoder = tokenizer.vocab_size
vocab_size_decoder = tokenizer.vocab_size

model = build_transformer(d_model, seq_len_encoder, seq_len_decoder, vocab_size_encoder, vocab_size_decoder)
model.to(device)

path_log = os.path.join("/home/wawa/pytorch-transformer/my_reproduce", strftime("%Y-%m-%d--%H:%M:%S", localtime()))
path_model = path_log+"/model/"

if not os.path.exists(path_log):
    os.makedirs(path_log)

logger = SummaryWriter(path_log)

# optimizer = AdamW(model.parameters(),lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-9)
# lr_scheduler = get_scheduler("linear",optimizer=optimizer,num_warmup_steps=0, num_training_steps=len(train_dataloader)*epoch_num)

test_loop(valid_dataloader, model)
# train_loop(train_dataloader,valid_dataloader, model, optimizer, epoch_num, logger, path_model)
# batch = next(iter(test_dataloader))
# batch_data = batch.to(device)
# batch_encoder_input = batch_data['input_ids']
# batch_encoder_mask = batch_data['attention_mask']
# batch_labels = batch_data['labels'].cpu().numpy() # [batch,seq]
# pred_tokens = greedy_decode(model, batch_encoder_input, batch_encoder_mask, max_target_len)

# pred_tokens_list = [pred_tokens[i,:] for i in range(pred_tokens.shape[0])]
# with tokenizer.as_target_tokenizer():
#     pred_sens = tokenizer.batch_decode(pred_tokens_list, skip_special_tokens=True)

# labels_tokens = np.where(batch_labels!=-100, batch_labels, tokenizer.pad_token_id)
# labels_tokens_list = [labels_tokens[i,:] for i in range(labels_tokens.shape[0])]
# label_sens = tokenizer.batch_decode(labels_tokens_list, skip_special_tokens=True)

# print(label_sens)
# print(pred_sens)

logger.close()



