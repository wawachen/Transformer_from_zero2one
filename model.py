from model_components import *

class Transformer(nn.Module):
    def __init__(self,d_model, seq_len_encoder, seq_len_decoder, vocab_size_encoder, vocab_size_decoder, dropout, hidden_dim, N, h, eps) -> None:
        super().__init__()
        self.encoder_embed = InputEmbeddings(d_model, vocab_size_encoder)
        self.decoder_embed = InputEmbeddings(d_model, vocab_size_decoder)

        self.encoder_pos = PositionEncoding(d_model, seq_len_encoder, dropout)
        self.decoder_pos = PositionEncoding(d_model, seq_len_decoder, dropout)

        self.encoder = Encoder(N, h, d_model, hidden_dim, dropout, eps)
        self.decoder = Decoder(N, d_model, h, dropout, hidden_dim, eps)

        self.projLayer = ProjectionLayer(d_model, vocab_size_decoder)

    def encode(self, x, src_mask):
        x = self.encoder(self.encoder_pos(self.encoder_embed(x)),src_mask)
        return x

    def decode(self,x, encode_out, src_mask, tgt_mask):
        x = self.decoder(self.decoder_pos(self.decoder_embed(x)),encode_out,src_mask,tgt_mask)
        return x

    def project(self, x):
        return self.projLayer(x)
    
def build_transformer(d_model, seq_len_encoder, seq_len_decoder, vocab_size_encoder, vocab_size_decoder, dropout=0.1, hidden_dim=2048, N=6, h=8, eps=10**-6):
    transformer = Transformer(d_model, seq_len_encoder, seq_len_decoder, vocab_size_encoder, vocab_size_decoder, dropout, hidden_dim, N, h, eps)

    # for param in transformer.parameters():
    #     if param.dim()>1:
    #         nn.init.xavier_uniform_(param)
    
    return transformer 




    







