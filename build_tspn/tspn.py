import torch
import torch.nn as nn
import torch.nn.functional as F

from fspool import FSEncoder
from transformer import Encoder
from set_prior import SetPrior
from size_predictor import SizePredictor

class TSPN(nn.Module):
    def __init__(self, encoder_latent, encoder_out, fspool_n_pieces, transformer_layers, transformer_attn_size, transformer_n_heads, n_ele_features, size_pred_width, pad_value, max_set_size):
        super(TSPN, self).__init__()

        self.pad_value = pad_value
        self.encoder = Encoder(transformer_layers, transformer_attn_size, transformer_n_heads, n_ele_features)
        self.n_ele_features = n_ele_features

        self._prior = SetPrior(n_ele_features)

        self._encoder = FSEncoder(in_channels=n_ele_features, encoder_dim=encoder_latent, encoder_output_channels=encoder_out, n_pieces=fspool_n_pieces)

        self._transformer = Encoder(transformer_layers, transformer_attn_size, transformer_n_heads, encoder_out)

        self._set_prediction = nn.Conv1d(in_channels=transformer_attn_size, out_channels=n_ele_features, kernel_size=1, bias=True)

        if self._set_prediction.bias is not None:
            nn.init.constant_(self._set_prediction.bias, 0.5)
        if self._set_prediction.weight is not None:
            nn.init.zeros_(self._set_prediction.weight)

        self._size_predictor = SizePredictor(in_features=transformer_attn_size, hidden_size=size_pred_width, max_units=max_set_size)

    def forward(self, initial_set, sampled_set, sizes):
        encoded = self._encoder(initial_set, self.n_ele_features, None, None)

        encoded_shaped= encoded.unsqueeze(1).repeat(1, self.max_set_size, 1)

        sampled_elements_conditioned = torch.cat([sampled_set, encoded_shaped], dim=2)


        mask = torch.arange(self.max_set_size, device=sizes.device)[None, :] >= sizes[:, None]
        mask = mask.float()

        pred_set_latent = self._transformer(sampled_elements_conditioned, mask)

        pred_set_latent= pred_set_latent.transpose(1, 2)

        pred_set = self._set_prediction(pred_set_latent)

        pred_set = pred_set.transpose(1, 2)

        return pred_set
    
    def sample_prior(self, sizes):
        total_elements = torch.sum(sizes)
        dist = self._prior(total_elements)
        return dist.sample()
    
    def sample_prior_batch(self, sizes):
        sampled_elements = self.sample_prior(sizes)
        padded = torch.full(
            (sizes.shape[0], self.max_set_size, self.num_element_features),
            self.pad_value,
            device=sizes.device
        )

        start = 0
        for i, s in enumerate(sizes.tolist()):
            end = start + s
            padded[i, :s, :] = sampled_elements[start:end]
            start = end
        return padded
    
    def encode_set(self, input_set, sizes):
        encoded = self._encoder(input_set, sizes)
        return encoded
    
    def predict_size(self, embedding):
        sizes = self._size_predictor(embedding)
        sizes = torch.softmax(sizes, dim=-1)
        return sizes
    
    def get_autoencoder_weights(self):
        return list(self._encoder.parameters()) + list(self._set_prediction.parameters()) + list(self._size_predictor.parameters())
    
    def get_prior_weights(self):
        return list(self._prior.parameters())
    
    def get_size_predictor_weights(self):
        return list(self._size_predictor.parameters())
    
    def prior_log_prob(self, initial_flat, sampled_set):
        batch_size = sampled_set.size(0)
        feature_dim = sampled_set.size(1)

        prior_dist = self._prior(batch_size)

        log_prob = prior_dist.log_prob(sampled_set).mean(dim=1)  # [batch_size]

        loss = -log_prob.mean()

        return log_prob, loss




    def pad_samples(self, samples, sizes, max_size, element_size, pad_value=0.0):
        batch_size = samples.size(0)
        padded = torch.full((batch_size, max_size, element_size),
                            pad_value, device=samples.device, dtype=samples.dtype)
        mask = torch.zeros((batch_size, max_size), device=samples.device, dtype=torch.bool)

        for i, size in enumerate(sizes):
            n = min(int(size.item()), max_size)
            padded[i, :n] = samples[i, :n]
            mask[i, :n] = True

        return padded, mask

