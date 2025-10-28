import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from model._modules import LayerNorm, FeedForward, MultiHeadAttention
from model._abstract_model import SequentialRecModel
from torch_geometric.nn import GATConv
import numpy as np

from pytorch_wavelets import DWT1DForward, DWT1DInverse

# ------------------------------
# 1. WaveletFilterBankLayer
# ------------------------------

class WaveletFilterBankLayer(nn.Module):
    """
    A GPU-accelerated layer for multi-level 1D Discrete Wavelet Transform (DWT)
    and Inverse DWT (IDWT).
    
    This layer decomposes an input sequence [B, L, H] into low-frequency (cA)
    and high-frequency (cD) components, adaptively enhances the high-frequency
    details, and then reconstructs the signal.
    """
    def __init__(self, args):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.wavelet_name = args.filter_type
        self.levels = args.dwt_levels
        self.device = args.device
        # self.padding_mode = args.padding_mode

        self.dwt = DWT1DForward(J=self.levels, wave=self.wavelet_name).to(self.device)
        self.idwt = DWT1DInverse(wave=self.wavelet_name).to(self.device)

        self.gain_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 4),
                nn.ReLU(),
                nn.Linear(self.hidden_size // 4, 1),
                nn.Sigmoid()
            ).to(self.device)
            for _ in range(self.levels)
        ])
        
        self.min_gain = args.min_gain
        self.max_gain = args.max_gain

    def enhance_detail(self, cD, level):

        cD_transposed = cD.permute(0, 2, 1)        
        cD_var = cD_transposed.var(dim=1)        
        gain_raw = self.gain_predictors[level](cD_var)        
        gain = self.min_gain + (self.max_gain - self.min_gain) * gain_raw        
        cD_enhanced = cD * gain.unsqueeze(-1)
        return cD_enhanced

    def forward(self, x):
        B, L, H = x.shape
        
        x_dwt_in = x.permute(0, 2, 1)        
        yl, yh = self.dwt(x_dwt_in)        
        yh_enhanced = []
        for level in range(self.levels):
            yh_enhanced.append(self.enhance_detail(yh[level], level))            
        recon_dwt_out = self.idwt((yl, yh_enhanced))        
        recon = recon_dwt_out.permute(0, 2, 1)        
        if recon.shape[1] != L:
            recon = recon[:, :L, :]            
        recon = recon + x        
        
        return recon

# ------------------------------
# 2. SocialGraphEncoder
# ------------------------------
class SocialGraphEncoder(nn.Module):
    """
    Encodes the social graph using a Graph Attention Network (GAT).
    
    It learns a graph embedding for *all* users in the system.
    These embeddings are then looked up by user_id in the main model.
    """
    def __init__(self, args, trust_graph_data):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_users = args.num_users
        self.num_layers = args.social_layers
        
        self.user_embedding = nn.Embedding(self.num_users, self.hidden_size)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        
        self.layers = nn.ModuleList()
        in_channels = self.hidden_size
        heads = args.num_attention_heads
        
        self.layers.append(GATConv(in_channels, self.hidden_size, heads=heads, dropout=args.hidden_dropout_prob, add_self_loops=False, edge_dim=1))
        in_channels = self.hidden_size * heads
        
        for _ in range(self.num_layers - 1):
            self.layers.append(GATConv(in_channels, self.hidden_size, heads=heads, dropout=args.hidden_dropout_prob, add_self_loops=False, edge_dim=1))
            in_channels = self.hidden_size * heads
            
        self.layers.append(GATConv(in_channels, self.hidden_size, heads=1, dropout=args.hidden_dropout_prob, add_self_loops=False, edge_dim=1))
        
        self.register_buffer('edge_index', trust_graph_data['edge_index'])
        self.register_buffer('edge_weight', trust_graph_data['edge_weight'].view(-1,1))

    def forward(self):
        x = self.user_embedding.weight
        
        for i, layer in enumerate(self.layers):
            x = F.dropout(x, p=self.dropout.p, training=self.training)
            x = layer(x, self.edge_index, edge_attr=self.edge_weight)
            
            if i < len(self.layers) - 1:
                x = F.elu(x)
                
        return x

# ------------------------------
# 3. Block
# ------------------------------
class Block(nn.Module):
    """
    A single block for the SWRecEncoder, analogous to a Transformer block.
    
    It consists of:
    1. A core GatedSWRecLayer (Self-Attention + Wavelet fusion)
    2. A standard FeedForward network
    (Assumes LayerNorm and Residuals are handled within these sub-modules)
    """
    def __init__(self, args):
        super().__init__()
        self.layer = GatedSWRecLayer(args) 
        self.feed_forward = FeedForward(args)

    def forward(self, hidden_states, attention_mask, social_emb):
        layer_output = self.layer(hidden_states, attention_mask, social_emb)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output


class GatedSWRecLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_size = args.hidden_size
        
        self.self_attention = MultiHeadAttention(args)
        self.wavelet_layer = WaveletFilterBankLayer(args)

        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, attention_mask, social_emb):
        att_out = self.self_attention(input_tensor, attention_mask)
        wav_out = self.wavelet_layer(input_tensor)

        fusion_input = torch.cat([att_out, wav_out], dim=-1)
        gate_feat = self.gate_mlp(fusion_input)

        gate_w = self.sigmoid(gate_feat.mean(dim=1))
        gate_w = gate_w.unsqueeze(1)

        fused_output = gate_w * wav_out + (1 - gate_w) * att_out
        
        return fused_output

# ------------------------------
# 5. Encoder
# ------------------------------
class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        block = Block(args)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, social_emb, output_all_encoded_layers=False):
        all_encoder_layers = [hidden_states]
        
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states, attention_mask, social_emb)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
                
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
            
        return all_encoder_layers

# ------------------------------
# 6. SoWaSARecModel
# ------------------------------
class SoWaSARecModel(SequentialRecModel):
    def __init__(self, args, trust_graph_data):
        super().__init__(args)
        self.device = args.device
        self.hidden_size = args.hidden_size
        
        self.item_encoder = Encoder(args)
        
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        
        if trust_graph_data is None:
            raise ValueError("trust_graph_data is required for SoWaSARecModel.")
            
        max_user_id = trust_graph_data['edge_index'].max().item()
        args.num_users = max(args.num_users, max_user_id + 1)
        
        self.social_encoder = SocialGraphEncoder(args, trust_graph_data)        
        self.social_beta = nn.Parameter(torch.tensor(0.3))

        self.apply(self.init_weights)

    def forward(self, input_ids, user_ids, all_sequence_output=False):
        seq_emb = self.add_position_embedding(input_ids)
        attn_mask = self.get_attention_mask(input_ids)
        
        all_social_embs = self.social_encoder()
        batch_social_embs = all_social_embs[user_ids]

        item_encoded_layers = self.item_encoder(seq_emb, 
                                                attn_mask, 
                                                batch_social_embs, 
                                                output_all_encoded_layers=True)
        
        if all_sequence_output:
            return item_encoded_layers
        
        sequence_output = item_encoded_layers[-1]
        sequence_output = sequence_output[:, -1, :]

        beta = torch.sigmoid(self.social_beta) 
        final_output = sequence_output + beta * batch_social_embs
        
        return final_output

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        seq_output = self.forward(input_ids, user_ids)
        item_emb = self.item_embeddings.weight
        logits = torch.matmul(seq_output, item_emb.transpose(0, 1))
        loss = nn.CrossEntropyLoss()(logits, answers)
        return loss

    def predict(self, input_ids, user_ids):
        return self.forward(input_ids, user_ids)