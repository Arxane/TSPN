import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model%n_heads==0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.d_model = d_model
        self.depth = d_model // n_heads

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)

        self.ff = nn.Linear(d_model, d_model, bias=False)


        nn.init.xavier_uniform_(self.wk.weight)
        nn.init.xavier_uniform_(self.wq.weight)
        nn.init.xavier_uniform_(self.wv.weight)
        nn.init.xavier_uniform_(self.ff.weight)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.n_heads, self.depth)
        return x.transpose(1,2)
    
    def forwad(self, v, k, q, mask=None):
        batch_size = q.size(0)

        Q = self.wq(q)
        K = self.wk(k)
        V = self.wv(v)

        Q = self.split_heads(q, batch_size)
        K = self.split_heads(k, batch_size)
        V = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = scaled_attention.transpose(1,2).contiguous()
        concat_attention = scaled_attention.view(batch_size, -1, self.d_model)

        output = self.ff(concat_attention)

        return output, attention_weights
    

class PointwiseFeedForward(nn.Module):
    def __init__(self, d_model):
        super(PointwiseFeedForward, self).__init__()
        self.ff1 = nn.Linear(d_model, d_model)
        self.ff2 = nn.Linear(d_model, d_model)

        nn.init.xavier_uniform_(self.ff1.weight)
        nn.init.xavier_uniform_(self.ff2.weight)
        nn.init.zeros_(self.ff1.bias)
        nn.init.zeros_(self.ff2.bias)

    def forward(self,x):
        x = F.leaky_relu(self.ff1(x))
        x = self.ff2(x)
        return x
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ffn = PointwiseFeedForward(d_model)
        self.ln1 = nn.LayerNorm(d_model, eps=1e-6)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, mask=None):
        attn_out, _  = self.mha(x, x, x, mask)
        out1 = self.ln1(x+attn_out)
        ffn_out = self.ffn(out1)
        out2 = self.ln2(out1 + ffn_out)
        return out2
    
class Encoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, input_channels):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        self.embedding = nn.Conv1d(in_channels=input_channels, out_channels=d_model, kernel_size=1, bias=True)

        nn.init.xavier_uniform_(self.embedding.weight)
        if self.embedding.bias is not None:
            nn.init.constant_(self.embedding.bias, 0.1)

        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, n_heads) for _ in range(n_layers)])

    def forward(self ,x, mask):
        if mask is not None and mask.dim()==2:
            mask = mask.unsqueeze(1).unsqueeze(1)


            x = self.embedding(x.transpose(1, 2))
            x = x.transpose(1,2)

            for i in range(self.n_layers):
                x = self.enc_layers[i](x, mask)
            return x
        
def scaled_dot_product_attention(q, k, v, mask=None):
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))
    dk = q.size(-1)

    scaled_attention_logits = matmul_qk / torch.sqrt(dk)

    if mask is not None:
        scaled_attention_logits = scaled_attention_logits.masked_fill(mask==0, float('-1e9'))

    attention_weights = F.softmax(scaled_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, v)
    return output, attention_weights

