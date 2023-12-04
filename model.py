
import torch
import torch.nn as nn
import numpy as np 
import torch.nn.functional as F



def patchify(images, patch_size):
     n,c, h, w = images.shape

    # Calcola il numero di patch orizzontali e verticali
     num_horizontal_patches = w // patch_size
     num_vertical_patches = h // patch_size

    # Dividi l'immagine in patch utilizzando un reshape
     patches = images.view(1, patch_size**2, num_horizontal_patches*num_vertical_patches)
     return patches


def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result


# Definisci l'encoder e il decoder del Transformer
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward):
        super(TransformerEncoder, self).__init__()

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward),
            num_encoder_layers
        )
    
    def forward(self, src):
        return self.encoder(src)

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_decoder_layers, dim_feedforward):
        super(TransformerDecoder, self).__init__()
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward),
            num_decoder_layers
        )
    
    def forward(self, src, memory):
        return self.decoder(src, memory)
    



class VisionTransformer(nn.Module):
    def __init__(self, chw, n_patches=16, n_heads=8, hidden_d = 512):
        super(VisionTransformer, self).__init__()

        self.chw = chw   # ( C , H , W )
        self.n_patches = n_patches
        self.n_heads = n_heads
        self.hidden_d = hidden_d

        assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])

        self.pos_embed = nn.Parameter(torch.tensor(get_positional_embeddings(n_patches**2, self.hidden_d)))
        self.pos_embed.requires_grad = False
        
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)
    

        self.cnn = nn.Conv2d(1, 1, kernel_size=5)
        self.encoder = TransformerEncoder(self.hidden_d, n_heads, num_encoder_layers = 2, dim_feedforward = 2048)
        self.decoder = TransformerDecoder(self.hidden_d, n_heads, num_decoder_layers = 2, dim_feedforward = 2048)

        self.back_to_input_size = nn.Linear(self.hidden_d, self.input_d)
        self.cnn_2 = nn.Conv2d(1, 1, kernel_size=5, stride=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cnn_3 = nn.Conv2d(1, 1, kernel_size=3, stride=2)
        self.Linear = nn.Linear(21, 1)
        self.activaction = nn.Sigmoid()
    
    def forward(self, src, memory):
        # Passa il batch di frame attraverso la CNN
        n, c, h, w = src.shape
        patches = patchify(src, self.n_patches)

        
        tokens = self.linear_mapper(patches)

        pos_embed = self.pos_embed.repeat(n, 1, 1)
        out = tokens + pos_embed

        
        # Passa l'output della CNN all'encoder del Transformer
        encoder_output = self.encoder(out)

        #patched_memory = patchify(memory, self.n_patches)
        
        #patched_memory = self.linear_mapper(patched_memory)
        #patched_memory = self.cnn(patched_memory)
        decoder_output = self.decoder(encoder_output, memory)
        #current_memory = self.back_to_input_size(decoder_output)

        memory = memory + decoder_output

        output = self.cnn_2(decoder_output)
        output = self.pool(output)
        output = self.cnn_3(output)
        output = self.pool(output)
        output = self.cnn_3(output)
        output = self.pool(output)
        output = output.view(-1)
        
        output = self.Linear(output)
        output = self.activaction(output)

        return output, memory