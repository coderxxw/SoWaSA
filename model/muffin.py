
import torch
import torch.nn as nn
import torch.nn.functional as F
from model._abstract_model import SequentialRecModel
from model._modules import LayerNorm
from model.module import LFMEncoder, GFMEncoder
import copy
import math
    
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish, 'silu':F.silu}


class MuffinConfig:
    """
    Configuration class that supports both attribute and dictionary-style access.
    This is needed because module.py uses both args.xxx and config['key'] access patterns.
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self._dict = kwargs
    
    def __getitem__(self, key):
        return self._dict[key]
    
    def __contains__(self, key):
        return key in self._dict


class MuffinModel(SequentialRecModel):
    """
    Muffin: Multi-Frequency Filtering Network for Sequential Recommendation
    
    This model uses Local Frequency Modulation (LFM) and Global Frequency Modulation (GFM)
    to capture both local and global frequency patterns in user behavior sequences.
    """
    def __init__(self, args, trust_graph_data=None):
        super(MuffinModel, self).__init__(args)
        
        # Model hyperparameters
        self.alpha = args.alpha
        self.beta = args.beta
        self.n_layers = args.num_hidden_layers
        self.hidden_size = args.hidden_size
        self.inner_size = args.inner_size
        self.hidden_dropout_prob = args.hidden_dropout_prob
        self.hidden_act = args.hidden_act
        self.initializer_range = args.initializer_range
        self.kernel_size = args.kernel_size
        
        # Build config object for encoders (supports both attribute and dict access)
        self.config = MuffinConfig(
            hidden_size=args.hidden_size,
            inner_size=args.inner_size,
            hidden_dropout_prob=args.hidden_dropout_prob,
            hidden_act=args.hidden_act,
            n_layers=args.num_hidden_layers,
            kernel_size=args.kernel_size,
            num_bands=args.num_bands,
            freq_dropout_prob=args.freq_dropout_prob,
            conv_layers=args.conv_layers,
            MAX_ITEM_LIST_LENGTH=args.max_seq_length,
        )
        
        # LFM and GFM encoders
        self.lfm_encoder = LFMEncoder(self.config)
        self.gfm_encoder = GFMEncoder(self.config)
        
        # Override item_embeddings from parent class
        self.item_embeddings = nn.Embedding(args.item_size, self.hidden_size, padding_idx=0)
        self.concat_layer = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)

        # UAF: Unified Adaptive Filter
        self.freq_conv_encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                padding_mode='reflect'
            ),
            nn.BatchNorm1d(self.hidden_size),
        )
        self.LayerNorm = LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.loss_fct = nn.CrossEntropyLoss()

        self.apply(self.init_weights)

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)
    
    def sequence_mask(self, input_ids):
        """Generate sequence mask for padding tokens"""
        mask = (input_ids != 0) * 1
        return mask.unsqueeze(-1) 
    
    def make_embedding(self, sequence, seq_mask):
        """Create item embeddings with masking, layer norm, and dropout"""
        # 安全检查：确保索引在有效范围内
        max_idx = sequence.max().item()
        if max_idx >= self.item_embeddings.num_embeddings:
            raise ValueError(
                f"输入序列中存在越界的 item id: {max_idx}，"
                f"但 embedding 表大小只有 {self.item_embeddings.num_embeddings}。"
                f"请检查数据预处理或增大 item_size。"
            )
        item_embeddings = self.item_embeddings(sequence)
        item_embeddings *= seq_mask
        item_embeddings = self.LayerNorm(item_embeddings)
        item_embeddings = self.dropout(item_embeddings)
        return item_embeddings
    
    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        """
        Forward pass of the Muffin model
        
        Args:
            input_ids: Input sequence of item ids [batch_size, seq_len]
            user_ids: User ids (not used in this model)
            all_sequence_output: Whether to return all sequence outputs
            
        Returns:
            output: Final output embedding [batch_size, hidden_size]
            gfm_output: GFM branch output [batch_size, hidden_size]
            lfm_output: LFM branch output [batch_size, hidden_size]
            total_lb_loss: Total load balancing loss
        """
        seq_mask = self.sequence_mask(input_ids)
        sequence_emb = self.make_embedding(input_ids, seq_mask)
        
        # Calculate sequence lengths (number of non-padding items)
        # Use clamp to ensure index is at least 0 (for sequences that are all padding)
        item_seq_len = (input_ids != 0).sum(dim=1)
        gather_index = (item_seq_len - 1).clamp(min=0)

        # UAF: Unified Adaptive Filter in frequency domain
        frequency_emb = torch.fft.rfft(sequence_emb, dim=1, norm='ortho')
        filter = torch.sigmoid(self.freq_conv_encoder(frequency_emb.abs().permute(0, 2, 1)))
        
        # GFM: Global Frequency Modulation
        gfm_layer = self.gfm_encoder(sequence_emb, seq_mask, filter, output_all_encoded_layers=True)
        gfm_output = gfm_layer[-1]
        gfm_output = self.gather_indexes(gfm_output, gather_index)

        # LFM: Local Frequency Modulation
        item_encoded_layers, total_lb_loss = self.lfm_encoder(sequence_emb, seq_mask, filter, output_all_encoded_layers=True)
        lfm_output = item_encoded_layers[-1]
        lfm_output = self.gather_indexes(lfm_output, gather_index)
        
        # Concatenate and fuse LFM and GFM outputs
        concat_output = torch.cat((lfm_output, gfm_output), dim=-1)
        output = self.concat_layer(concat_output)

        # Residual connection with last hidden state
        last_hidden_state = self.gather_indexes(sequence_emb, gather_index)
        output = self.LayerNorm(output + last_hidden_state)
        output = self.dropout(output)
        
        return output, gfm_output, lfm_output, total_lb_loss

     
    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        """
        Calculate training loss
        
        Args:
            input_ids: Input sequence [batch_size, seq_len]
            answers: Positive item ids [batch_size]
            neg_answers: Negative item ids (not used)
            same_target: Same target mask (not used)
            user_ids: User ids (not used)
            
        Returns:
            loss: Total loss including main loss, auxiliary losses, and load balancing loss
        """
        seq_output, gfm_output, lfm_output, total_lb_loss = self.forward(input_ids, user_ids)
        
        test_item_emb = self.item_embeddings.weight
        
        # Main loss
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss_fct(logits, answers)
        
        # Auxiliary loss for GFM branch
        logits = torch.matmul(gfm_output, test_item_emb.transpose(0, 1))
        gfm_loss = self.loss_fct(logits, answers)
        
        # Auxiliary loss for LFM branch
        logits = torch.matmul(lfm_output, test_item_emb.transpose(0, 1))
        lfm_loss = self.loss_fct(logits, answers) 
        
        # Total loss = main loss + auxiliary losses + load balancing loss
        loss = loss + self.alpha * (gfm_loss + lfm_loss)
        loss += self.beta * total_lb_loss
        
        return loss
    
    def predict(self, input_ids, user_ids=None, all_sequence_output=False):
        """
        Predict - returns sequence output for compatibility with trainer
        
        Args:
            input_ids: Input sequence [batch_size, seq_len]
            user_ids: User ids (not used)
            all_sequence_output: Whether to return all sequence outputs
            
        Returns:
            seq_output: Sequence output embedding [batch_size, hidden_size]
                       (trainer will multiply with item embeddings)
        """
        seq_output, _, _, _ = self.forward(input_ids, user_ids)
        # Return seq_output instead of scores, trainer's predict_full will do the matmul
        return seq_output

    def init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm) or isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv1d):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()