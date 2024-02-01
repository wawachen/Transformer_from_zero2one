import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super(InputEmbeddings, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        #the paper addressed the embedding should be multiplied by square root of the d_model
        return self.embedding(x) * math.sqrt(self.d_model)
        
class PositionEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout:float):
        super(PositionEncoding, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout) # for overfitting
        
        pos_emb = torch.zeros(seq_len, d_model)

        pos_term = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)
        denomi_term = torch.exp(-torch.arange(0,d_model,2).float() * math.log(10000) / self.d_model)
        
        #right alignment: equal or has 1 dimension
        # (seq_len,1)*(d_model)--> （seq_len,1）* (1,d_model)--> (seq_len,d_model)*(seq_len,d_model)-->  (seq_len,d_model) auto-broacast
        pos_emb[:, 0::2] = torch.sin(pos_term * denomi_term)
        pos_emb[:, 1::2] = torch.cos(pos_term * denomi_term)

        pos_emb = pos_emb.unsqueeze(0)  # (1,seq_len,d_model)
        #look into https://blog.csdn.net/weixin_38145317/article/details/104917218
        self.register_buffer('pe', pos_emb)
        
    def forward(self, x):
        x = x+(self.pe[:,:x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)
        
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model:int, h: int, dropout:float):
        super(MultiHeadAttentionBlock, self).__init__()
        
        self.d_model = d_model
        self.w_q = nn.Linear(d_model,d_model,bias=False)
        self.w_k = nn.Linear(d_model,d_model,bias=False)
        self.w_v = nn.Linear(d_model, d_model,bias=False)
        self.w_o = nn.Linear(d_model, d_model,bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.head_dim = d_model // h
        self.h = h
        assert (d_model % h == 0)

    def attention(self, query, key, value, mask, dropout: nn.Dropout):
        #(batch,h,seq,h_dim)x(batch,h,h_dim，seq)--> (batch,h,seq,seq)
        d_k = query.shape[-1]
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # print(attention_scores.shape, mask.shape)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
            
        attention_prob=attention_scores.softmax(dim=-1)
        
        if dropout is not None:
            attention_prob = dropout(attention_prob)
        
        return torch.matmul(attention_prob,value),attention_prob

    def forward(self, query, key, value, mask):
        q = self.w_q(query) # (batch,seq,d_model)
        k = self.w_k(key) # (batch,seq,d_model)
        v = self.w_v(value) # (batch,seq,d_model)

        q_h = q.contiguous().view(q.shape[0], q.shape[1], self.h, self.head_dim)  #(batch,seq,h,h_dim)
        k_h = k.contiguous().view(k.shape[0], k.shape[1], self.h, self.head_dim)  #(batch,seq,h,h_dim)
        v_h = v.contiguous().view(v.shape[0], v.shape[1], self.h, self.head_dim)  #(batch,seq,h,h_dim)
        
        q_t = q_h.transpose(1,2) #(batch,h,seq,h_dim)
        k_t = k_h.transpose(1,2) #(batch,h,seq,h_dim)
        v_t = v_h.transpose(1,2)  #(batch,h,seq,h_dim)
        
        #(batch,h,seq,h_dim)-->(batch,seq,h,h_dim)->(batch,seq,d_model)
        x = self.attention(q_t, k_t, v_t, mask, self.dropout)[0].transpose(1, 2).contiguous().view(q.shape[0], q.shape[1], self.d_model)
        
        # (batch,seq,d_model)
        return self.w_o(x)

#layernormalisation we can call torch.nn.LayerNorm(normalised_shape,eps,bias=)
class Feedforward_network(nn.Module):
    def __init__(self, d_model:int, hidden_dim: int, dropout: float):
        super(Feedforward_network, self).__init__()
        self.layer1 = nn.Linear(d_model,hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, d_model)
        self.ac1 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        return self.layer2(self.dropout(self.ac1(self.layer1(x))))

class Residual_connection(nn.Module):
    def __init__(self, d_model: int, dropout: float, eps:float) -> None:
        super(Residual_connection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x, sublayer):
        #identity x and output y
        return x+self.dropout(sublayer(self.norm(x)))

class Encoder_block(nn.Module):
    def __init__(self, h: int, d_model: int, hidden_dim: int, dropout: float, eps: float):
        super(Encoder_block, self).__init__()
        self.attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        self.f_net = Feedforward_network(d_model, hidden_dim, dropout)
        self.residual_connections = nn.ModuleList([Residual_connection(d_model, dropout, eps) for _ in range(2)])
    
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.f_net)
        return x

class Encoder(nn.Module):
    def __init__(self, N: int, h: int, d_model: int, hidden_dim: int, dropout: float, eps: float):
        super(Encoder, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model, eps=eps)
        self.encoder_blocks = nn.ModuleList([Encoder_block(h, d_model, hidden_dim, dropout, eps) for _ in range(N)])
        
    def forward(self,x,mask):
        for layer in self.encoder_blocks:
            x = layer(x,mask)

        return self.layer_norm(x) 

class decoder_blocks(nn.Module):
    def __init__(self, d_model:int, h:int, dropout:float, hidden_dim:int, eps:float) :
        super().__init__()
        self.attention_blocks = nn.ModuleList([MultiHeadAttentionBlock(d_model, h, dropout) for _ in range(2)])
        self.f_net = Feedforward_network(d_model, hidden_dim, dropout)
        self.residual_connections = nn.ModuleList([Residual_connection(d_model, dropout, eps) for _ in range(3)])

    def forward(self, x, encode_out, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.attention_blocks[0](x,x,x,tgt_mask))
        x = self.residual_connections[1](x, lambda x : self.attention_blocks[1](x,encode_out,encode_out,src_mask))
        x = self.residual_connections[2](x, self.f_net)

        return x

class Decoder(nn.Module):
    def __init__(self, N:int, d_model:int, h:int, dropout:float, hidden_dim:int, eps:float):
        super().__init__()
        self.decoder_blocks = nn.ModuleList([decoder_blocks(d_model, h, dropout, hidden_dim, eps) for _ in range(N)])
        self.norm =  nn.LayerNorm(d_model, eps=eps)

    def forward(self,x,encode_out,src_mask, tgt_mask):
        for layer in self.decoder_blocks:
            x = layer(x, encode_out, src_mask, tgt_mask)

        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.net = nn.Linear(d_model, vocab_size)
        self.ac = nn.Softmax(dim=-1)

    def forward(self,x):
        return self.ac(self.net(x))



        
        
        
