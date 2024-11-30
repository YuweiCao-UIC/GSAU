import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType
from recbole.model.layers import TransformerEncoder

def u_i_neg_item_sampling(pos_item_embeddings):
    # Generate negative item embeddings for negative U-I embedding pairs by shuffling item embeddings
    while True:
        perm_indices = torch.randperm(pos_item_embeddings.size(0))
        u_i_neg_item_embeddings = pos_item_embeddings[perm_indices]
            
        # Check for mismatches: no original index should match the shuffled index
        if not (perm_indices == torch.arange(pos_item_embeddings.size(0))).any():
            break  # If no matches, break the loop
    # neg: (pos_seq_embeddings, u_i_neg_item_embeddings)
    return u_i_neg_item_embeddings

def gather_indexes(output, gather_index):
    """
    Gathers the vectors at the specific positions over a minibatch.
    Needed by the sequential encoder for prediction (to get the embedding of the last item in the embedded sequence).
    gather_index: a tensor of shape [B 1], containing the indices of the last items.
    """
    gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
    output_tensor = output.gather(dim=1, index=gather_index)
    return output_tensor.squeeze(1)

class GSAU(GeneralRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # training loss
        self.enable_u_i_uniformity = config['enable_u_i_uniformity']
        self.rec = config['rec']
        self.gamma = config['gamma']

        # perdiction
        self.predict_with_graph_encoder = config['predict_with_graph_encoder']

        self.enable_graph_encoder = config['enable_graph_encoder']
        self.enable_sequential_encoder = config['enable_sequential_encoder']
        # at least one encoder needs to be enabled
        assert self.enable_graph_encoder or self.enable_sequential_encoder
        if self.predict_with_graph_encoder:
            assert self.enable_graph_encoder
        else:
            self.enable_sequential_encoder

        # embedding layer
        self.embedding_size = config['embedding_size']
        self.embedding_layer = EmbeddingLayer(n_users=self.n_users, n_items=self.n_items, embedding_size=self.embedding_size)

        # graph encoder
        if self.enable_graph_encoder:
            self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
            self.norm_adj = self.get_norm_adj_mat().to(self.device)
            self.n_layers = config['n_layers']
            self.align_per_layer = config['align_per_layer']
            self.graph_encoder = LightGCNEncoder(self.embedding_layer, self.n_users, self.n_items, self.embedding_size, self.norm_adj, self.n_layers, self.align_per_layer)
        
        # sequential encoder
        if self.enable_sequential_encoder:
            self.sequential_encoder = config['sequential_encoder']
            assert self.sequential_encoder in ['BERT4Rec', 'SASRec']
            self.sequential_encoder_name = self.sequential_encoder
            # SequentialRecommender configs
            self.ITEM_SEQ = self.ITEM_ID + config["LIST_SUFFIX"]
            self.ITEM_SEQ_LEN = config["ITEM_LIST_LENGTH_FIELD"]
            self.loss_fct = nn.CrossEntropyLoss()
            if self.sequential_encoder_name == 'BERT4Rec':
                self.max_seq_length = config["MAX_ITEM_LIST_LENGTH"]
                self.MASK_ITEM_SEQ = config["MASK_ITEM_SEQ"]
                self.MASK_INDEX = config["MASK_INDEX"]
                self.POS_ITEMS = config["POS_ITEMS"]
                self.sequential_encoder = BERT4RecEncoder(embedding_layer=self.embedding_layer, config=config, item_num=self.n_items, max_seq_length=self.max_seq_length)
            else:
                self.POS_ITEM_ID = self.ITEM_ID
                self.sequential_encoder = SASRecEncoder(embedding_layer=self.embedding_layer, config=config)

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def get_norm_adj_mat(self):
        # build adj matrix
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()

        # Update A directly by iterating through the entries
        for i, j in zip(inter_M.row, inter_M.col):
            A[i, j + self.n_users] = 1
        for i, j in zip(inter_M_t.row, inter_M_t.col):
            A[i + self.n_users, j] = 1
        
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def forward(self, graph_user=None, graph_item=None, masked_seq=None, masked_index=None, seq_items=None, item_seq=None, item_seq_len=None, pos_items=None):
        if self.enable_graph_encoder:
            assert graph_user is not None
            assert graph_item is not None
            all_layers_graph_user_e, all_layers_graph_item_e = self.graph_encoder(graph_user, graph_item)
            if not self.enable_sequential_encoder: # only graph
                return [F.normalize(graph_user_e, dim=-1) for graph_user_e in all_layers_graph_user_e], \
                    [F.normalize(graph_item_e, dim=-1) for graph_item_e in all_layers_graph_item_e]
        if self.enable_sequential_encoder:
            if self.sequential_encoder_name == 'BERT4Rec':
                assert masked_seq is not None
                assert masked_index is not None
                assert seq_items is not None
                pos_seq_embeddings, pos_item_embeddings = self.sequential_encoder(masked_seq, masked_index, seq_items)
            else: # SASRec
                assert item_seq is not None
                assert item_seq_len is not None
                assert pos_items is not None
                pos_seq_embeddings, pos_item_embeddings = self.sequential_encoder(item_seq, item_seq_len, pos_items)
            if not self.enable_graph_encoder: # only sequential
                return F.normalize(pos_seq_embeddings, dim=-1), F.normalize(pos_item_embeddings, dim=-1)
        return [F.normalize(graph_user_e, dim=-1) for graph_user_e in all_layers_graph_user_e], \
            [F.normalize(graph_item_e, dim=-1) for graph_item_e in all_layers_graph_item_e], \
            F.normalize(pos_seq_embeddings, dim=-1), \
            F.normalize(pos_item_embeddings, dim=-1)

    @staticmethod
    def alignment(x, y, alpha=2):
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()

    @staticmethod
    def uniformity(x, y=None, t=2):
        if y is None:
            # consider all possible pairs in x
            loss = torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
        else:
            # consider (x, y) pairs
            loss = (x - y).norm(p=2, dim=1).pow(2).mul(-t).exp().mean().log()
        return loss


    def calculate_loss(self, graph_interaction, sequential_interaction):
        # prepare data
        # graph
        if self.enable_graph_encoder:
            graph_user = graph_interaction[self.USER_ID]
            graph_item = graph_interaction[self.ITEM_ID]
        # sequential
        if self.enable_sequential_encoder:
            if self.sequential_encoder_name == 'BERT4Rec':
                masked_seq = sequential_interaction[self.MASK_ITEM_SEQ]
                masked_index = sequential_interaction[self.MASK_INDEX] # [B max_mask_item_length], padded with 0, max_mask_item_length=int(mask_ratio * max_seq_length), see https://github.com/RUCAIBox/RecBole/blob/master/recbole/data/transform.py for details
                seq_items = sequential_interaction[self.POS_ITEMS] # [B max_mask_item_length], padded with 0
            else: # SASRec
                item_seq = sequential_interaction[self.ITEM_SEQ]
                item_seq_len = sequential_interaction[self.ITEM_SEQ_LEN]
                pos_items = sequential_interaction[self.POS_ITEM_ID]
    
        # forward to get embeddings
        if self.enable_graph_encoder and self.enable_sequential_encoder:
            if self.sequential_encoder_name == 'BERT4Rec':
                all_layers_graph_user_e, all_layers_graph_item_e, pos_seq_embeddings, pos_item_embeddings = self.forward(graph_user=graph_user, graph_item=graph_item, masked_seq=masked_seq, masked_index=masked_index, seq_items=seq_items)
            else: # SASRec
                all_layers_graph_user_e, all_layers_graph_item_e, pos_seq_embeddings, pos_item_embeddings = self.forward(graph_user=graph_user, graph_item=graph_item, item_seq=item_seq, item_seq_len=item_seq_len, pos_items=pos_items)
        elif self.enable_graph_encoder:
            all_layers_graph_user_e, all_layers_graph_item_e = self.forward(graph_user=graph_user, graph_item=graph_item)
        else:
            if self.sequential_encoder_name == 'BERT4Rec':
                pos_seq_embeddings, pos_item_embeddings = self.forward(masked_seq=masked_seq, masked_index=masked_index, seq_items=seq_items)
            else: # SASRec
                pos_seq_embeddings, pos_item_embeddings = self.forward(item_seq=item_seq, item_seq_len=item_seq_len, pos_items=pos_items)
        
        # calculate loss
        # graph loss
        if self.enable_graph_encoder:
            # graph alignment loss
            graph_align = None
            if self.align_per_layer:
                for graph_user_e, graph_item_e in zip(all_layers_graph_user_e[1:], all_layers_graph_item_e[1:]):
                    a_i2u_loss = self.alignment(all_layers_graph_user_e[0], graph_item_e)
                    graph_align = (a_i2u_loss if graph_align is None else graph_align + a_i2u_loss)
                    a_u2i_loss = self.alignment(all_layers_graph_item_e[0], graph_user_e)
                    graph_align = graph_align + a_u2i_loss
                graph_align /= (2 * (len(all_layers_graph_user_e) - 1))
            else:
                # align summary
                graph_align = self.alignment(all_layers_graph_user_e[-1], all_layers_graph_item_e[-1])
            # graph uniformity loss
            if self.align_per_layer:
                graph_uniform = self.gamma * (self.uniformity(all_layers_graph_user_e[0]) + self.uniformity(all_layers_graph_item_e[0]))
            else:
                # unify summary
                graph_uniform = self.gamma * (self.uniformity(all_layers_graph_user_e[-1]) + self.uniformity(all_layers_graph_item_e[-1]))

        # sequential loss
        if self.enable_sequential_encoder:
            # sequential alignment loss
            if not self.rec:
                # use alignment loss, (pos_seq_embeddings, pos_item_embeddings)
                sequential_align = self.alignment(pos_seq_embeddings, pos_item_embeddings)
            else:
                # use CE loss
                if self.sequential_encoder_name == 'BERT4Rec':
                    test_item_emb = self.sequential_encoder.embedding.get_item_embeddings([i for i in range(self.n_items)]) # delete masked token, [item_num H]
                    logits = torch.matmul(pos_seq_embeddings, test_item_emb.transpose(0, 1)) # (num_masked_items_in_batch, item_num)
                    # Update the mask
                    masked_index = masked_index > 0
                    masked_index = masked_index.view(-1) # (B*mask_len,)
                    # Update seq_items
                    seq_items = seq_items.view(-1) # (B, mask_len) -> (B*mask_len,)
                    pos_items = torch.masked_select(seq_items, masked_index) # (num_masked_items_in_batch,)
                    sequential_align = self.loss_fct(logits, pos_items)
                else: # SASRec
                    test_item_emb = self.sequential_encoder.item_embedding.weight
                    logits = torch.matmul(pos_seq_embeddings, test_item_emb.transpose(0, 1))
                    sequential_align = self.loss_fct(logits, pos_items)

            # sequential uniformity loss
            if self.enable_u_i_uniformity:
                # neg: all possible pairs in (pos_seq_embeddings), all possible pairs in (pos_item_embeddings), (pos_seq_embeddings, sequential_u_i_neg_item_embeddings)
                sequential_u_i_neg_item_embeddings = u_i_neg_item_sampling(pos_item_embeddings)
                sequential_uniform = self.gamma * (self.uniformity(pos_seq_embeddings) + self.uniformity(pos_item_embeddings) + self.uniformity(pos_seq_embeddings, sequential_u_i_neg_item_embeddings))
            else:
                # neg: all possible pairs in (pos_seq_embeddings), all possible pairs in (pos_item_embeddings)
                sequential_uniform = self.gamma * (self.uniformity(pos_seq_embeddings) + self.uniformity(pos_item_embeddings))

        # overall loss
        if self.enable_graph_encoder and self.enable_sequential_encoder:
            return graph_align + graph_uniform + sequential_align + sequential_uniform
        elif self.enable_graph_encoder:
            return graph_align + graph_uniform
        else:
            return sequential_align + sequential_uniform

    def full_sort_predict(self, interaction):
        if self.predict_with_graph_encoder:
            # predict with graph_encoder
            user_id = interaction[self.USER_ID]
            all_layers_user_all_embeddings, all_layers_item_all_embeddings = self.graph_encoder.get_all_embeddings()
            user_all_embeddings = all_layers_user_all_embeddings[-1]
            item_all_embeddings = all_layers_item_all_embeddings[-1]
            user_e = user_all_embeddings[user_id]
            score = torch.matmul(user_e, item_all_embeddings.transpose(0, 1))
            return score.view(-1)
        else:
            # predict with sequential_encoder
            item_seq = interaction[self.ITEM_SEQ]
            item_seq_len = interaction[self.ITEM_SEQ_LEN]
            if self.sequential_encoder_name == 'BERT4Rec':
                item_seq = self.sequential_encoder.reconstruct_test_data(item_seq, item_seq_len)
                seq_output = self.sequential_encoder._embed_item_seq(item_seq)
                seq_output = gather_indexes(seq_output, item_seq_len - 1)  # [B H]
                test_items_emb = self.sequential_encoder.embedding.get_item_embeddings([i for i in range(self.n_items)]) # delete masked token
            else: # SASRec
                seq_output = self.sequential_encoder._embed_item_seq(item_seq, item_seq_len)
                test_items_emb = self.sequential_encoder.item_embedding.weight
            scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1)) # [B, item_num]
            return scores

class EmbeddingLayer(nn.Module):
    def __init__(self, n_users, n_items, embedding_size):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_size, padding_idx=0)
        self.item_embedding = nn.Embedding(n_items, embedding_size, padding_idx=0)
    
    def get_item_embeddings(self):
        return self.item_embedding.weight
    
    def get_user_embeddings(self):
        return self.user_embedding.weight

class LightGCNEncoder(nn.Module):
    def __init__(self, embedding_layer, user_num, item_num, emb_size, norm_adj, n_layers=3, align_per_layer=True):
        super().__init__()
        self.n_users = user_num
        self.n_items = item_num
        self.n_layers = n_layers
        self.norm_adj = norm_adj
        self.embedding_layer = embedding_layer
        self.align_per_layer = align_per_layer
    
    def get_ego_embeddings(self):
        user_embeddings = self.embedding_layer.get_user_embeddings()
        item_embeddings = self.embedding_layer.get_item_embeddings()
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings
    
    def get_all_embeddings(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        all_layers_user_all_embeddings, all_layers_item_all_embeddings = [],[]
        init_user_all_embeddings, init_item_all_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])
        all_layers_user_all_embeddings.append(init_user_all_embeddings)
        all_layers_item_all_embeddings.append(init_item_all_embeddings)
        
        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj, all_embeddings)
            embeddings_list.append(all_embeddings)
        # get per-layer user and item embeddings
        if self.align_per_layer:
            for each in embeddings_list[1:]:
                layer_user_all_embeddings, layer_item_all_embeddings = torch.split(each, [self.n_users, self.n_items])
                all_layers_user_all_embeddings.append(layer_user_all_embeddings)
                all_layers_item_all_embeddings.append(layer_item_all_embeddings)
        # get summarized user and item embeddings
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        all_layers_user_all_embeddings.append(user_all_embeddings)
        all_layers_item_all_embeddings.append(item_all_embeddings)
        return all_layers_user_all_embeddings, all_layers_item_all_embeddings

    def forward(self, user_id, item_id):
        all_layers_user_all_embeddings, all_layers_item_all_embeddings = self.get_all_embeddings()
        all_layers_u_embed, all_layers_i_embed = [], []
        for i in range(len(all_layers_user_all_embeddings)):
            user_all_embeddings = all_layers_user_all_embeddings[i]
            item_all_embeddings = all_layers_item_all_embeddings[i]
            u_embed = user_all_embeddings[user_id]
            i_embed = item_all_embeddings[item_id]
            all_layers_u_embed.append(u_embed)
            all_layers_i_embed.append(i_embed)
        return all_layers_u_embed, all_layers_i_embed

class BERT4RecEmbedding(nn.Module):
    def __init__(self, embedding_layer, embed_size, max_seq_length):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.token_mask = nn.Parameter(torch.Tensor(1, embed_size))
        self.position_embedding = nn.Embedding(max_seq_length, embed_size)
    
    def forward(self, item_seq):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        item_emb = self.get_item_embeddings(item_seq)
        input_emb = item_emb + position_embedding
        return input_emb

    def get_item_embeddings(self, item_seq):
        return torch.cat((self.embedding_layer.get_item_embeddings(), self.token_mask), 0)[item_seq]

class BERT4RecEncoder(nn.Module):
    def __init__(self, embedding_layer, config, item_num, max_seq_length):
        super().__init__()
        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config[
            "inner_size"
        ]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.mask_ratio = config["mask_ratio"]

        self.MASK_ITEM_SEQ = config["MASK_ITEM_SEQ"]
        self.POS_ITEMS = config["POS_ITEMS"]
        self.NEG_ITEMS = config["NEG_ITEMS"]
        self.MASK_INDEX = config["MASK_INDEX"]

        self.initializer_range = config["initializer_range"]

        # load dataset info
        self.mask_token = item_num
        self.mask_item_length = int(self.mask_ratio * max_seq_length)

        # define layers
        self.embedding = BERT4RecEmbedding(embedding_layer, self.hidden_size, max_seq_length)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.output_ffn = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_gelu = nn.GELU()
        self.output_ln = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        
    @staticmethod
    def get_attention_mask(item_seq, bidirectional=False):
        """
        Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention.
        Needed by the sequential encoder.
        """
        attention_mask = item_seq != 0
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(
                extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1))
            )
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask

    @staticmethod
    def multi_hot_embed(masked_index, max_length):
        """
        For memory, we only need calculate loss for masked position.
        Generate a multi-hot vector to indicate the masked position for masked sequence, and then is used for
        gathering the masked position hidden representation.

        Examples:
            sequence: [1 2 3 4 5]

            masked_sequence: [1 mask 3 mask 5]

            masked_index: [1, 3]

            max_length: 5

            multi_hot_embed: [[0 1 0 0 0], [0 0 0 1 0]]
        """
        masked_index = masked_index.view(-1)
        multi_hot = torch.zeros(
            masked_index.size(0), max_length, device=masked_index.device
        )
        multi_hot[torch.arange(masked_index.size(0)), masked_index] = 1
        return multi_hot

    def reconstruct_test_data(self, item_seq, item_seq_len):
        """
        Add mask token at the last position according to the lengths of item_seq
        """
        padding = torch.zeros(
            item_seq.size(0), dtype=torch.long, device=item_seq.device
        )  # [B]
        item_seq = torch.cat((item_seq, padding.unsqueeze(-1)), dim=-1)  # [B max_len+1]
        for batch_id, last_position in enumerate(item_seq_len):
            item_seq[batch_id][last_position] = self.mask_token
        item_seq = item_seq[:, 1:]
        return item_seq

    def _embed_item_seq(self, item_seq):
        # embed item_seq (EmbeddingLayer item embeddings + position embedding)
        input_emb = self.embedding(item_seq)

        # layer norm and dropout
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        # transformer
        # wouldn't pay attention to PADs
        extended_attention_mask = self.get_attention_mask(item_seq, bidirectional=True)
        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )

        # feed forward, nonlinear, and layer norm
        ffn_output = self.output_ffn(trm_output[-1])
        ffn_output = self.output_gelu(ffn_output)
        output = self.output_ln(ffn_output) # [B L H]

        return output

    def forward(self, item_seq, masked_index, seq_items):
        # item_seq: [item_id_1, mask_id, item_id_2, mask_id, item_id_3, ...]
        batch_size = masked_index.size(0)
        mask_len = masked_index.size(1)

        # BERT4Rec encoding of the item_seq
        output = self._embed_item_seq(item_seq)
        
        # get BERT4Rec encoding of the masked items in item_seq
        pred_index_map = self.multi_hot_embed(
            masked_index, item_seq.size(-1)
        )  # [B*mask_len max_len]
        # [B mask_len] -> [B mask_len max_len] multi hot
        pred_index_map = pred_index_map.view(
            batch_size, mask_len, -1
        )  # [B mask_len max_len]
        # [B mask_len max_len] * [B max_len H] -> [B mask_len H]
        # only calculate loss for masked position
        seq_masked_items_e = torch.bmm(pred_index_map, output)  # [B mask_len H]

        # get the EmbeddingLayer embeddings of the items in the sequences
        masked_items_e = self.embedding.get_item_embeddings(seq_items) # [B mask_len H]

        # Reshape the embeddings to 2D
        seq_masked_items_e = seq_masked_items_e.view(-1, self.hidden_size) # [B*mask_len H]
        masked_items_e = masked_items_e.view(-1, self.hidden_size) # [B*mask_len H]
        # Update the mask
        masked_index = masked_index > 0
        masked_index = masked_index.view(-1) # (B*mask_len,)

        # Genarate positive U-I embedding pairs
        pos_seq_embeddings = torch.masked_select(seq_masked_items_e, masked_index.unsqueeze(-1)).view(-1, self.hidden_size)
        pos_item_embeddings = torch.masked_select(masked_items_e, masked_index.unsqueeze(-1)).view(-1, self.hidden_size)
        
        return pos_seq_embeddings, pos_item_embeddings


class SASRecEncoder(nn.Module):
    def __init__(self, embedding_layer, config):
        super().__init__()
        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config[
            "inner_size"
        ]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.initializer_range = config["initializer_range"]

        self.max_seq_length = config["MAX_ITEM_LIST_LENGTH"]

        # define layers
        self.item_embedding = embedding_layer.item_embedding
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @staticmethod
    def get_attention_mask(item_seq, bidirectional=False):
        """
        Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention.
        Needed by the sequential encoder.
        """
        attention_mask = item_seq != 0
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(
                extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1))
            )
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask

    def _embed_item_seq(self, item_seq, item_seq_len):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = gather_indexes(output, item_seq_len - 1)
        return output  # [B H]
    
    def forward(self, item_seq, item_seq_len, pos_items):
        seq_e = self._embed_item_seq(item_seq, item_seq_len)
        items_e = self.item_embedding(pos_items) # [B H]
        return seq_e, items_e
    