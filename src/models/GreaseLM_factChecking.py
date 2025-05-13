## this code is modified version of GreaseLM model https://github.com/snap-stanford/GreaseLM.git for fact checking problem

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertForMaskedLM, RobertaModel
from transformers.models.bert.modeling_bert import BertEncoder
#from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.data import Data
from tqdm.notebook import tqdm
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from tqdm.notebook import tqdm
from torch_geometric.graphgym import optim
from torch.nn.utils import clip_grad_norm_
import numpy as np
from lightning.pytorch.utilities import CombinedLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops  #, softmax
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter
import random
import pandas as pd


def gelu(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


class TypedLinear(nn.Linear):
    def __init__(self, in_features, out_features, n_type):
        super().__init__(in_features, n_type * out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.n_type = n_type

    def forward(self, X, type_ids=None):
        """
        X: tensor of shape (*, in_features)
        type_ids: long tensor of shape (*)
        """
        output = super().forward(X)
        if type_ids is None:
            return output
        output_shape = output.size()[:-1] + (self.out_features,)
        output = output.view(-1, self.n_type, self.out_features)
        idx = torch.arange(output.size(0), dtype=torch.long, device=type_ids.device)
        output = output[idx, type_ids.view(-1)].view(*output_shape)
        return output


class MLP(nn.Module):
    """
    Multi-layer perceptron

    Parameters
    ----------
    num_layers: number of hidden layers
    """
    activation_classes = {'gelu': GELU, 'relu': nn.ReLU, 'tanh': nn.Tanh}

    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, batch_norm=False,
                 init_last_layer_bias_to_zero=False, layer_norm=False, activation='gelu'):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm

        assert not (self.batch_norm and self.layer_norm)

        self.layers = nn.Sequential()
        for i in range(self.num_layers + 1):
            n_in = self.input_size if i == 0 else self.hidden_size
            n_out = self.hidden_size if i < self.num_layers else self.output_size
            self.layers.add_module(f'{i}-Linear', nn.Linear(n_in, n_out))
            if i < self.num_layers:
                self.layers.add_module(f'{i}-Dropout', nn.Dropout(self.dropout))
                if self.batch_norm:
                    self.layers.add_module(f'{i}-BatchNorm1d', nn.BatchNorm1d(self.hidden_size))
                if self.layer_norm:
                    self.layers.add_module(f'{i}-LayerNorm', nn.LayerNorm(self.hidden_size))
                self.layers.add_module(f'{i}-{activation}', self.activation_classes[activation.lower()]())
        if init_last_layer_bias_to_zero:
            self.layers[-1].bias.data.fill_(0)

    def forward(self, input):
        return self.layers(input)


class MaxPoolLayer(nn.Module):
    """
    A layer that performs max pooling along the sequence dimension
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, mask_or_lengths):
        """
        inputs: tensor of shape (batch_size, seq_len, hidden_size)
        mask_or_lengths: tensor of shape (batch_size) or (batch_size, seq_len)

        returns: tensor of shape (batch_size, hidden_size)
        """
        bs, sl, _ = inputs.size()
        if len(mask_or_lengths.size()) == 1:
            mask = (torch.arange(sl, device=inputs.device).unsqueeze(0).expand(bs, sl) >= mask_or_lengths.unsqueeze(1))
        else:
            mask = mask_or_lengths
        masked_inputs = inputs.masked_fill(mask.unsqueeze(-1).expand_as(inputs), float('-inf'))
        max_pooled = masked_inputs.max(1)[0]
        return max_pooled


class MeanPoolLayer(nn.Module):
    """
    A layer that performs mean pooling along the sequence dimension
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, mask_or_lengths):
        """
        inputs: tensor of shape (batch_size, seq_len, hidden_size)
        mask_or_lengths: tensor of shape (batch_size) or (batch_size, seq_len)

        returns: tensor of shape (batch_size, hidden_size)
        """
        bs, sl, _ = inputs.size()
        if len(mask_or_lengths.size()) == 1:
            mask = (torch.arange(sl, device=inputs.device).unsqueeze(0).expand(bs, sl) >= mask_or_lengths.unsqueeze(1))
            lengths = mask_or_lengths.float()
        else:
            mask, lengths = mask_or_lengths, (1 - mask_or_lengths.float()).sum(1)
        masked_inputs = inputs.masked_fill(mask.unsqueeze(-1).expand_as(inputs), 0.0)
        mean_pooled = masked_inputs.sum(1) / lengths.unsqueeze(-1)
        return mean_pooled


def dropout_mask(x, sz, p: float):
    """
    Return a dropout mask of the same type as `x`, size `sz`, with probability `p` to cancel an element.

    (adapted from https://github.com/fastai/fastai/blob/1.0.42/fastai/text/models/awd_lstm.py)
    """
    return x.new(*sz).bernoulli_(1 - p).div_(1 - p)


class EmbeddingDropout(nn.Module):
    """
    Apply dropout with probabily `embed_p` to an embedding layer `emb`.

    (adapted from https://github.com/fastai/fastai/blob/1.0.42/fastai/text/models/awd_lstm.py)
    """

    def __init__(self, emb: nn.Module, embed_p: float):
        super().__init__()
        self.emb, self.embed_p = emb, embed_p
        self.pad_idx = self.emb.padding_idx
        if self.pad_idx is None:
            self.pad_idx = -1

    def forward(self, words):
        if self.training and self.embed_p != 0:
            size = (self.emb.weight.size(0), 1)
            mask = dropout_mask(self.emb.weight.data, size, self.embed_p)
            masked_embed = self.emb.weight * mask
        else:
            masked_embed = self.emb.weight
        return F.embedding(words, masked_embed, self.pad_idx, self.emb.max_norm,
                           self.emb.norm_type, self.emb.scale_grad_by_freq, self.emb.sparse)


class RNNDropout(nn.Module):
    "Dropout with probability `p` that is consistent on the seq_len dimension."

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.:
            return x
        m = dropout_mask(x.data, (x.size(0), 1, x.size(2)), self.p)
        return x * m


class LSTMEncoder(nn.Module):

    def __init__(self, vocab_size=300, emb_size=300, hidden_size=300, num_layers=2, bidirectional=True,
                 emb_p=0, input_p=0, hidden_p=0, output_p=0, pretrained_emb=None, pooling=True, pad=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.emb_p = emb_p
        self.input_p = input_p
        self.hidden_p = hidden_p
        self.output_p = output_p
        self.pooling = pooling

        self.emb = EmbeddingDropout(nn.Embedding(vocab_size, emb_size), emb_p)
        if pretrained_emb is not None:
            self.emb.emb.weight.data.copy_(pretrained_emb)
        else:
            bias = np.sqrt(6.0 / emb_size)
            nn.init.uniform_(self.emb.emb.weight, -bias, bias)
        self.input_dropout = nn.Dropout(input_p)
        self.output_dropout = nn.Dropout(output_p)
        self.rnn = nn.LSTM(input_size=emb_size, hidden_size=(hidden_size // 2 if self.bidirectional else hidden_size),
                           num_layers=num_layers, dropout=hidden_p, bidirectional=bidirectional,
                           batch_first=True)
        self.max_pool = MaxPoolLayer()

    def forward(self, inputs, lengths):
        """
        inputs: tensor of shape (batch_size, seq_len)
        lengths: tensor of shape (batch_size)

        returns: tensor of shape (batch_size, hidden_size)
        """
        bz, full_length = inputs.size()
        embed = self.emb(inputs)
        embed = self.input_dropout(embed)
        lstm_inputs = pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=False)
        rnn_outputs, _ = self.rnn(lstm_inputs)
        rnn_outputs, _ = pad_packed_sequence(rnn_outputs, batch_first=True, total_length=full_length)
        rnn_outputs = self.output_dropout(rnn_outputs)
        return self.max_pool(rnn_outputs, lengths) if self.pooling else rnn_outputs


class TripleEncoder(nn.Module):
    def __init__(self, emb_dim, hidden_dim, input_p, output_p, hidden_p, num_layers, bidirectional=True, pad=False,
                 concept_emb=None, relation_emb=None
                 ):
        super().__init__()
        if pad:
            raise NotImplementedError
        self.input_p = input_p
        self.output_p = output_p
        self.hidden_p = hidden_p
        self.cpt_emb = concept_emb
        self.rel_emb = relation_emb
        self.input_dropout = nn.Dropout(input_p)
        self.output_dropout = nn.Dropout(output_p)
        self.bidirectional = bidirectional
        self.rnn = nn.GRU(input_size=emb_dim, hidden_size=(hidden_dim // 2 if self.bidirectional else hidden_dim),
                          num_layers=num_layers, dropout=hidden_p, bidirectional=bidirectional,
                          batch_first=True)

    def forward(self, inputs):
        '''
        inputs: (batch_size, seq_len)

        returns: (batch_size, h_dim(*2))
        '''
        bz, sl = inputs.size()
        h, r, t = torch.chunk(inputs, 3, dim=1)  # (bz, 1)

        h, t = self.input_dropout(self.cpt_emb(h)), self.input_dropout(self.cpt_emb(t))  # (bz, 1, dim)
        r = self.input_dropout(self.rel_emb(r))
        inputs = torch.cat((h, r, t), dim=1)  # (bz, 3, dim)
        rnn_outputs, _ = self.rnn(inputs)  # (bz, 3, dim)
        if self.bidirectional:
            outputs_f, outputs_b = torch.chunk(rnn_outputs, 2, dim=2)
            outputs = torch.cat((outputs_f[:, -1, :], outputs_b[:, 0, :]), 1)  # (bz, 2 * h_dim)
        else:
            outputs = rnn_outputs[:, -1, :]

        return self.output_dropout(outputs)


class MatrixVectorScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, v, mask=None):
        """
        q: tensor of shape (n*b, d_k)
        k: tensor of shape (n*b, l, d_k)
        v: tensor of shape (n*b, l, d_v)

        returns: tensor of shape (n*b, d_v), tensor of shape(n*b, l)
        """
        attn = (q.unsqueeze(1) * k).sum(2)  # (n*b, l)
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = (attn.unsqueeze(2) * v).sum(1)
        return output, attn


class AttPoolLayer(nn.Module):

    def __init__(self, d_q, d_k, dropout=0.1):
        super().__init__()
        self.w_qs = nn.Linear(d_q, d_k)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_q + d_k)))
        self.attention = MatrixVectorScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, mask=None):
        """
        q: tensor of shape (b, d_q)
        k: tensor of shape (b, l, d_k)
        mask: tensor of shape (b, l) (optional, default None)
        returns: tensor of shape (b, d_k)
        """
        qs = self.w_qs(q)  # (b, d_k)
        output, attn = self.attention(qs, k, k, mask=mask)
        output = self.dropout(output)
        return output, attn


class MultiheadAttPoolLayer(nn.Module):

    def __init__(self, n_head, d_q_original, d_k_original, dropout=0.1):
        super().__init__()
        assert d_k_original % n_head == 0  # make sure the outpute dimension equals to d_k_origin
        self.n_head = n_head
        self.d_k = d_k_original // n_head
        self.d_v = d_k_original // n_head

        self.w_qs = nn.Linear(d_q_original, n_head * self.d_k)
        self.w_ks = nn.Linear(d_k_original, n_head * self.d_k)
        self.w_vs = nn.Linear(d_k_original, n_head * self.d_v)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_q_original + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_v)))

        self.attention = MatrixVectorScaledDotProductAttention(temperature=np.power(self.d_k, 0.5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, mask=None):
        """
        q: tensor of shape (b, d_q_original)
        k: tensor of shape (b, l, d_k_original)
        mask: tensor of shape (b, l) (optional, default None)
        returns: tensor of shape (b, n*d_v)
        """
        n_head, d_k, d_v = self.n_head, self.d_k, self.d_v

        bs, _ = q.size()
        bs, len_k, _ = k.size()

        qs = self.w_qs(q).view(bs, n_head, d_k)  # (b, n, dk)
        ks = self.w_ks(k).view(bs, len_k, n_head, d_k)  # (b, l, n, dk)
        vs = self.w_vs(k).view(bs, len_k, n_head, d_v)  # (b, l, n, dv)

        qs = qs.permute(1, 0, 2).contiguous().view(n_head * bs, d_k)
        ks = ks.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_k)
        vs = vs.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1)
        output, attn = self.attention(qs, ks, vs, mask=mask)

        output = output.view(n_head, bs, d_v)
        output = output.permute(1, 0, 2).contiguous().view(bs, n_head * d_v)  # (b, n*dv)
        output = self.dropout(output)
        return output, attn


class TypedMultiheadAttPoolLayer(nn.Module):

    def __init__(self, n_head, d_q_original, d_k_original, dropout=0.1, n_type=1):
        super().__init__()
        assert d_k_original % n_head == 0  # make sure the outpute dimension equals to d_k_origin
        self.n_head = n_head
        self.d_k = d_k_original // n_head
        self.d_v = d_k_original // n_head

        self.w_qs = nn.Linear(d_q_original, n_head * self.d_k)
        self.w_ks = TypedLinear(d_k_original, n_head * self.d_k, n_type)
        self.w_vs = TypedLinear(d_k_original, n_head * self.d_v, n_type)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_q_original + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_v)))

        self.attention = MatrixVectorScaledDotProductAttention(temperature=np.power(self.d_k, 0.5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, mask=None, type_ids=None):
        """
        q: tensor of shape (b, d_q_original)
        k: tensor of shape (b, l, d_k_original)
        mask: bool tensor of shape (b, l) (optional, default None)
        type_ids: long tensor of shape (b, l) (optional, default None)
        returns: tensor of shape (b, n*d_v)
        """
        n_head, d_k, d_v = self.n_head, self.d_k, self.d_v

        bs, _ = q.size()
        bs, len_k, _ = k.size()

        qs = self.w_qs(q).view(bs, n_head, d_k)  # (b, n, dk)
        ks = self.w_ks(k, type_ids=type_ids).view(bs, len_k, n_head, d_k)  # (b, l, n, dk)
        vs = self.w_vs(k, type_ids=type_ids).view(bs, len_k, n_head, d_v)  # (b, l, n, dv)

        qs = qs.permute(1, 0, 2).contiguous().view(n_head * bs, d_k)
        ks = ks.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_k)
        vs = vs.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1)
        output, attn = self.attention(qs, ks, vs, mask=mask)

        output = output.view(n_head, bs, d_v)
        output = output.permute(1, 0, 2).contiguous().view(bs, n_head * d_v)  # (b, n*dv)
        output = self.dropout(output)
        return output, attn


class BilinearAttentionLayer(nn.Module):

    def __init__(self, query_dim, value_dim):
        super().__init__()
        self.linear = nn.Linear(value_dim, query_dim, bias=False)
        self.softmax = nn.Softmax(1)

    def forward(self, query, value, node_mask=None):
        """
        query: tensor of shape (batch_size, query_dim)
        value: tensor of shape (batch_size, seq_len, value_dim)
        node_mask: tensor of shape (batch_size, seq_len)

        returns: tensor of shape (batch_size, value_dim)
        """
        attn = self.linear(value).bmm(query.unsqueeze(-1))
        attn = self.softmax(attn.squeeze(-1))
        if node_mask is not None:
            attn = attn * node_mask
            attn = attn / attn.sum(1, keepdim=True)
        pooled = attn.unsqueeze(1).bmm(value).squeeze(1)
        return pooled, attn


def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   memory_efficient: bool = True,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.
    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # # To limit numerical errors from large vector elements outside the mask, we zero these out.
            # result = nn.functional.softmax(vector * mask, dim=dim)
            # result = result * mask
            # result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
            raise NotImplementedError
        else:
            masked_vector = vector.masked_fill(mask.to(dtype=torch.uint8), mask_fill_value)
            result = nn.functional.softmax(masked_vector, dim=dim)
            result = result * (1 - mask)
    return result


class DiffTopK(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, k):
        """
        x: tensor of shape (batch_size, n_node)
        k: int
        returns: tensor of shape (batch_size, n_node)
        """
        bs, _ = x.size()
        _, topk_indexes = x.topk(k, 1)  # (batch_size, k)
        output = x.new_zeros(x.size())
        ri = torch.arange(bs).unsqueeze(1).expand(bs, k).contiguous().view(-1)
        output[ri, topk_indexes.view(-1)] = 1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None


class SimilarityFunction(nn.Module):
    """
    A ``SimilarityFunction`` takes a pair of tensors with the same shape, and computes a similarity
    function on the vectors in the last dimension.  For example, the tensors might both have shape
    `(batch_size, sentence_length, embedding_dim)`, and we will compute some function of the two
    vectors of length `embedding_dim` for each position `(batch_size, sentence_length)`, returning a
    tensor of shape `(batch_size, sentence_length)`.
    The similarity function could be as simple as a dot product, or it could be a more complex,
    parameterized function.
    """
    default_implementation = 'dot_product'

    def forward(self, tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
        """
        Takes two tensors of the same shape, such as ``(batch_size, length_1, length_2,
        embedding_dim)``.  Computes a (possibly parameterized) similarity on the final dimension
        and returns a tensor with one less dimension, such as ``(batch_size, length_1, length_2)``.
        """
        raise NotImplementedError


class DotProductSimilarity(SimilarityFunction):
    """
    This similarity function simply computes the dot product between each pair of vectors, with an
    optional scaling to reduce the variance of the output elements.
    Parameters
    ----------
    scale_output : ``bool``, optional
        If ``True``, we will scale the output by ``math.sqrt(tensor.size(-1))``, to reduce the
        variance in the result.
    """

    def __init__(self, scale_output: bool = False) -> None:
        super(DotProductSimilarity, self).__init__()
        self._scale_output = scale_output

    def forward(self, tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
        result = (tensor_1 * tensor_2).sum(dim=-1)
        if self._scale_output:
            result *= math.sqrt(tensor_1.size(-1))
        return result


class MatrixAttention(nn.Module):
    def __init__(self, similarity_function: SimilarityFunction = None) -> None:
        super().__init__()
        self._similarity_function = similarity_function or DotProductSimilarity()

    def forward(self, matrix_1: torch.Tensor, matrix_2: torch.Tensor) -> torch.Tensor:
        tiled_matrix_1 = matrix_1.unsqueeze(2).expand(matrix_1.size()[0],
                                                      matrix_1.size()[1],
                                                      matrix_2.size()[1],
                                                      matrix_1.size()[2])
        tiled_matrix_2 = matrix_2.unsqueeze(1).expand(matrix_2.size()[0],
                                                      matrix_1.size()[1],
                                                      matrix_2.size()[1],
                                                      matrix_2.size()[2])

        return self._similarity_function(tiled_matrix_1, tiled_matrix_2)



import torch.nn as nn
import torch.nn.functional as F
from cycler import cycler
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import unbatch
from transformers import BertModel, BertForMaskedLM, RobertaModel, BertConfig, RobertaConfig
from transformers.models.bert.modeling_bert import BertEncoder
#from modeling import modeling_gnn
#from utils import layers
#from claimReview import kg_gat
#from modeling import modeling_gnn
#from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.data import Data, Batch
#import kg_gat
from tqdm.notebook import tqdm
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from tqdm.notebook import tqdm
from torch_geometric.graphgym import optim
from torch.nn.utils import clip_grad_norm_
import numpy as np
#from sentence_transformers import SentenceTransformer
from lightning.pytorch.utilities import CombinedLoader
import torch_geometric.transforms as T
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.nn import GATv2Conv
from transformers.modeling_tf_pytorch_utils import load_pytorch_checkpoint_in_tf2_model
import h5py
import logging
import os
from transformers.utils.hub import cached_file
from torch import optim
from transformers import PretrainedConfig
from transformers.file_utils import (
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    WEIGHTS_NAME,
    default_cache_path
)
logger = logging.getLogger(__name__)
ModelClass = RobertaModel

class RobertaPooler2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class LMGNN(nn.Module):

    def __init__(self, args={},model_name='roberta-large', k=5, n_ntype=4, n_etype=38,
                 concept_dim=384, n_attention_head=2,
                 fc_dim=200, n_fc_layer=0, p_emb=0.2, p_gnn=0.2, p_fc=0.2,
                 init_range=0.02, ie_dim=200, info_exchange=True, ie_layer_num=1, sep_ie_layers=True, layer_id=-1):
        super().__init__()
        config, _ = ModelClass.config_class.from_pretrained(
            "roberta-large",
            cache_dir=None, return_unused_kwargs=True,
            force_download=False,
            output_hidden_states=True
        )

        #self.init_range = init_range

        self.k = k
        self.concept_dim = concept_dim
        self.n_attention_head = n_attention_head
        self.activation =GELU()
        if k >= 0:
            self.pooler = MultiheadAttPoolLayer(n_attention_head, config.hidden_size, concept_dim)

        #concat_vec_dim = concept_dim + config.hidden_size. OLD
        concat_vec_dim = concept_dim + config.hidden_size
        self.fc = MLP(concat_vec_dim, fc_dim, 3, n_fc_layer, p_fc, layer_norm=True)

        self.dropout_e = nn.Dropout(p_emb)
        self.dropout_fc = nn.Dropout(p_fc)

        self.mp = TextKGMessagePassing.from_pretrained("roberta-large", output_hidden_states=True,
                                                                          args=args, k=k,
                                                                          n_ntype=n_ntype, n_etype=n_etype,
                                                                          dropout=p_gnn, concept_dim=concept_dim,
                                                                          ie_dim=ie_dim, p_fc=p_fc,
                                                                          info_exchange=info_exchange,
                                                                          ie_layer_num=ie_layer_num,
                                                                          sep_ie_layers=sep_ie_layers)

        self.layer_id = layer_id


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, graph, emb_data=None, cache_output=False):

        outputs, gnn_output,emb = self.mp(input_ids, attention_mask, graph, output_hidden_states=True)

        all_hidden_states = outputs[-1] # ([bs, seq_len, sent_dim] for _ in range(25))

        hidden_states = all_hidden_states[self.layer_id] # [bs, seq_len, sent_dim]
        sent_vecs = self.mp.pooler(hidden_states) # [bs, sent_dim]

        Z_vecs = gnn_output[:,0]   #(batch_size, dim_node)

        sent_vecs_for_pooler = sent_vecs
        graph_vecs, pool_attn = self.pooler(sent_vecs_for_pooler, gnn_output, mask=None)

        concat = torch.cat((sent_vecs, graph_vecs), 1)
        logits = self.fc(self.dropout_fc(concat))

        return logits, gnn_output, emb


class TextKGMessagePassing(RobertaModel):

    def __init__(self, config, k=5, n_ntype=4, n_etype=38, dropout=0.2, n_attention_head=2,concept_dim=384, ie_dim=200, p_fc=0.2, info_exchange=True, ie_layer_num=1, sep_ie_layers=False):
        super().__init__(config=config)

        self.n_ntype = n_ntype
        self.n_etype = n_etype

        self.hidden_size = concept_dim
        self.emb_node_type = nn.Linear(self.n_ntype, concept_dim // 2)

        self.basis_f = 'sin' #['id', 'linact', 'sin', 'none']
        if self.basis_f in ['id']:
            self.emb_score = nn.Linear(1, concept_dim // 2)
        elif self.basis_f in ['linact']:
            self.B_lin = nn.Linear(1, concept_dim // 2)
            self.emb_score = nn.Linear(concept_dim // 2, concept_dim // 2)
        elif self.basis_f in ['sin']:
            self.emb_score = nn.Linear(concept_dim // 2, concept_dim // 2)

        self.k = k

        self.Vh = nn.Linear(concept_dim, concept_dim)
        self.Vx = nn.Linear(concept_dim, concept_dim)

        self.activation = GELU()
        self.dropout = nn.Dropout(dropout)
        self.dropout_rate = dropout

        self.encoder = RoBERTaGAT(config, k=k, n_ntype=n_ntype, n_etype=n_etype, hidden_size=384, dropout=dropout, concept_dim=concept_dim, ie_dim=ie_dim, p_fc=p_fc, info_exchange=info_exchange, ie_layer_num=ie_layer_num, sep_ie_layers=sep_ie_layers)

        self.sent_dim = config.hidden_size


    def forward(self, input_ids, attention_mask, graph, cache_output=False, position_ids=None, head_mask=None, output_hidden_states=True):

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if len(attention_mask.size()) == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif len(attention_mask.size()) == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise ValueError("Attention mask should be either 1D or 2D.")

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids)
      
        H = graph.x.to(torch.float32)
        encoder_outputs, _X,emb = self.encoder(embedding_output,
                                           extended_attention_mask, head_mask,  graph.x.to(torch.float32), graph.edge_index, graph.edge_attr.to(torch.float32), graph.batch,
                                           output_hidden_states=output_hidden_states)

        # LM outputs
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        output_LM = (sequence_output, pooled_output,) + encoder_outputs[1:]


        # GNN outputs
        output = self.activation(self.Vh(H) + self.Vx(_X))
        output_GNN = self.dropout(output)

        return output_LM, output_GNN,emb

    @classmethod
    def from_pretrained(cls, pre_model, *model_args, **kwargs):

        config = RobertaConfig.from_pretrained(pre_model, **kwargs)

        resolved_archive_file = cached_file(pre_model, "pytorch_model.bin", force_download=True)

        model = cls(config, *model_args)

        state_dict = torch.load(resolved_archive_file, map_location="cuda:0")

        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if "gamma" in key:
                new_key = key.replace("gamma", "weight")
            if "beta" in key:
                new_key = key.replace("beta", "bias")
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        all_keys = list(state_dict.keys())

        def load(module: nn.Module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs,
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

            # Make sure we are able to load base models as well as derived models (with heads)

        start_prefix = ""
        model_to_load = model
        has_prefix_module = any(s.startswith(cls.base_model_prefix) for s in state_dict.keys())
        if not hasattr(model, cls.base_model_prefix) and has_prefix_module:
            start_prefix = cls.base_model_prefix + "."
        if hasattr(model, cls.base_model_prefix) and not has_prefix_module:
            model_to_load = getattr(model, cls.base_model_prefix)

        load(model_to_load, prefix=start_prefix)

        if model.__class__.__name__ != model_to_load.__class__.__name__:
            base_model_state_dict = model_to_load.state_dict().keys()
            head_model_state_dict_without_base_prefix = [
                key.split(cls.base_model_prefix + ".")[-1] for key in model.state_dict().keys()
            ]

            missing_keys.extend(head_model_state_dict_without_base_prefix - base_model_state_dict)

        model.tie_weights()
        model.eval()
        return model


class RoBERTaGAT(BertEncoder):

    def __init__(self, config, k=5, fc_dim=200, n_fc_layer=0,n_ntype=1, n_etype=38, hidden_size=384, dropout=0.2, concept_dim=384, ie_dim=200, p_fc=0.2, info_exchange=True, ie_layer_num=1, sep_ie_layers=False):
        super().__init__(config)

        self.k = k
        self.gnn_layers = nn.ModuleList([GNNStack(hidden_size, concept_dim, 384, 3,task='graph') for _ in range(k)])

        self.activation = GELU()
        self.dropout_rate = dropout
        self.dropout_fc = nn.Dropout(p_fc)
        self.sent_dim = config.hidden_size
        #self.sent_dim = concept_dim
        #self.sep_ie_layers = False
        self.sep_ie_layers = sep_ie_layers
        if sep_ie_layers:
            self.ie_layers = nn.ModuleList([MLP(self.sent_dim +concept_dim , ie_dim, self.sent_dim + concept_dim, ie_layer_num, p_fc) for _ in range(k)])
            #self.ie_layers = nn.ModuleList([layers.MLP(self.sent_dim  , ie_dim, self.sent_dim  , ie_layer_num, p_fc) for _ in range(k)])
        else:
            self.ie_layer = MLP(self.sent_dim+concept_dim , ie_dim, self.sent_dim+concept_dim , ie_layer_num, p_fc)

        self.concept_dim = concept_dim
        self.num_hidden_layers = 24
        self.info_exchange = info_exchange
        self.hidden_size = hidden_size
        self.concept_dim = concept_dim


    def forward(self, hidden_states, attention_mask, head_mask,  x, edge_index, edge_attr, batch, output_attentions=False, output_hidden_states=True):
        bs = hidden_states.size(0)
        all_hidden_states = ()
        all_attentions = ()
        head_mask = [None] * self.num_hidden_layers
        for i, layer_module in enumerate(self.layer):
            # LM
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
            if i >= self.num_hidden_layers - self.k:
                gnn_layer_index = i - self.num_hidden_layers + self.k
                x, emb = self.gnn_layers[gnn_layer_index](x, edge_index, edge_attr, batch)
                x = self.activation(x)
                x = F.dropout(x, self.dropout_rate, training = self.training)

                if self.info_exchange == True or (self.info_exchange == "every-other-layer" and (i - self.num_hidden_layers + self.k) % 2 == 0):
                    batch_arr= batch.cpu().detach().numpy()
                    temp = [0]
                    for i, elem in enumerate(batch_arr):
                        if (i < len(batch_arr) - 1 and elem != batch_arr[i + 1]):
                            temp.append(i + 1)
                    ref = torch.Tensor(temp).to(torch.int32).to('cuda')
                  
                    context_node_gnn_feats = torch.index_select(x, 0, ref)
                    context_node_lm_feats = hidden_states[:, 0, :]
                    context_node_feats = torch.cat([context_node_lm_feats, context_node_gnn_feats], dim=1)

                    if self.sep_ie_layers:
                        context_node_feats = self.ie_layers[gnn_layer_index](context_node_feats)
                    else:
                        context_node_feats = self.ie_layer(context_node_feats)
                    context_node_lm_feats, context_node_gnn_feats = torch.split(context_node_feats, [context_node_lm_feats.size(1), context_node_gnn_feats.size(1)], dim=1)
                    hidden_states[:, 0, :] = context_node_lm_feats
                    x[temp, :] =context_node_gnn_feats

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            outputs = outputs + (all_attentions,)

        batch_s =ref.size(dim=0)
        x_new= x.unsqueeze(0).expand(batch_s, -1, -1)
      
        return outputs, x_new,emb
