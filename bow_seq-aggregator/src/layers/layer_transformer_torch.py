import torch
import copy
import torch.nn.functional as F
# import torch.nn.Module as Module
import torch.nn as nn
# import _VF
from torch.nn.init import constant_

class TransformerEncoder(nn.Module):
	r"""TransformerEncoder is a stack of N encoder layers

	Args:
		encoder_layer: an instance of the TransformerEncoderLayer() class (required).
		num_layers: the number of sub-encoder-layers in the encoder (required).
		norm: the layer normalization component (optional).

	Examples::
		# >>> encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
		# >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
	"""

	def __init__(self, encoder_layer, num_layers, norm=None):
		super(TransformerEncoder, self).__init__()
		self.layers = _get_clones(encoder_layer, num_layers)
		self.num_layers = num_layers
		self.norm = norm

	def forward(self, src, mask=None, src_key_padding_mask=None):
		r"""Pass the input through the endocder layers in turn.

		Args:
			src: the sequnce to the encoder (required).
			mask: the mask for the src sequence (optional).
			src_key_padding_mask: the mask for the src keys per batch (optional).

		Shape:
			see the docs in Transformer class.
		"""
		output = src

		for i in range(self.num_layers):
			output = self.layers[i](output, src_mask=mask,
									src_key_padding_mask=src_key_padding_mask)

		if self.norm:
			output = self.norm(output)

		return output

class TransformerEncoderLayer(nn.Module):
	r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
	This standard encoder layer is based on the paper "Attention Is All You Need".
	Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
	Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
	Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
	in a different way during application.

	Args:
		d_model: the number of expected features in the input (required).
		nhead: the number of heads in the multiheadattention models (required).
		dim_feedforward: the dimension of the feedforward network model (default=2048).
		dropout: the dropout value (default=0.1).

	Examples::
		# >>> encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
	"""

	def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
		super(TransformerEncoderLayer, self).__init__()
		self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
		# Implementation of Feedforward model
		self.linear1 = nn.Linear(d_model, dim_feedforward)
		self.dropout = nn.Dropout(dropout)
		self.linear2 = nn.Linear(dim_feedforward, d_model)

		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)

	def forward(self, src, src_mask=None, src_key_padding_mask=None):
		r"""Pass the input through the endocder layer.

		Args:
			src: the sequnce to the encoder layer (required).
			src_mask: the mask for the src sequence (optional).
			src_key_padding_mask: the mask for the src keys per batch (optional).

		Shape:
			see the docs in Transformer class.
		"""
		src2 = self.self_attn(src, src, src, attn_mask=src_mask,
							  key_padding_mask=src_key_padding_mask)[0]
		src = src + self.dropout1(src2)
		src = self.norm1(src)
		src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
		src = src + self.dropout2(src2)
		src = self.norm2(src)
		return src


def _get_clones(module, N):
	return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MultiheadAttention(nn.Module):
	r"""Allows the model to jointly attend to information
	from different representation subspaces.
	See reference: Attention Is All You Need

	.. math::
		\text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
		\text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

	Args:
		embed_dim: total dimension of the model.
		num_heads: parallel attention heads.
		dropout: a Dropout layer on attn_output_weights. Default: 0.0.
		bias: add bias as module parameter. Default: True.
		add_bias_kv: add bias to the key and value sequences at dim=0.
		add_zero_attn: add a new batch of zeros to the key and
					   value sequences at dim=1.
		kdim: total number of features in key. Default: None.
		vdim: total number of features in key. Default: None.

		Note: if kdim and vdim are None, they will be set to embed_dim such that
		query, key, and value have the same number of features.

	Examples::

		# >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
		# >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
	"""

	def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
		super(MultiheadAttention, self).__init__()
		self.embed_dim = embed_dim
		self.kdim = kdim if kdim is not None else embed_dim
		self.vdim = vdim if vdim is not None else embed_dim
		self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

		self.num_heads = num_heads
		self.dropout = dropout
		self.head_dim = embed_dim // num_heads
		assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

		self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))

		if self._qkv_same_embed_dim is False:
			self.q_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
			self.k_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.kdim))
			self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.vdim))

		if bias:
			self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
		else:
			self.register_parameter('in_proj_bias', None)
		self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

		if add_bias_kv:
			self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim))
			self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim))
		else:
			self.bias_k = self.bias_v = None

		self.add_zero_attn = add_zero_attn

		self._reset_parameters()

	def _reset_parameters(self):
		if self._qkv_same_embed_dim:
			nn.init.xavier_uniform_(self.in_proj_weight)
		else:
			nn.init.xavier_uniform_(self.q_proj_weight)
			nn.init.xavier_uniform_(self.k_proj_weight)
			nn.init.xavier_uniform_(self.v_proj_weight)

		if self.in_proj_bias is not None:
			constant_(self.in_proj_bias, 0.)
			constant_(self.out_proj.bias, 0.)
		if self.bias_k is not None:
			nn.init.xavier_normal_(self.bias_k)
		if self.bias_v is not None:
			nn.init.xavier_normal_(self.bias_v)

	def forward(self, query, key, value, key_padding_mask=None,
				need_weights=True, attn_mask=None):
		r"""
	Args:
		query, key, value: map a query and a set of key-value pairs to an output.
			See "Attention Is All You Need" for more details.
		key_padding_mask: if provided, specified padding elements in the key will
			be ignored by the attention. This is an binary mask. When the value is True,
			the corresponding value on the attention layer will be filled with -inf.
		need_weights: output attn_output_weights.
		attn_mask: mask that prevents attention to certain positions. This is an additive mask
			(i.e. the values will be added to the attention layer).

	Shape:
		- Inputs:
		- query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
		  the embedding dimension.
		- key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
		  the embedding dimension.
		- value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
		  the embedding dimension.
		- key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
		- attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.

		- Outputs:
		- attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
		  E is the embedding dimension.
		- attn_output_weights: :math:`(N, L, S)` where N is the batch size,
		  L is the target sequence length, S is the source sequence length.
		"""
		if hasattr(self, '_qkv_same_embed_dim') and self._qkv_same_embed_dim is False:
			return multi_head_attention_forward(
				query, key, value, self.embed_dim, self.num_heads,
				self.in_proj_weight, self.in_proj_bias,
				self.bias_k, self.bias_v, self.add_zero_attn,
				self.dropout, self.out_proj.weight, self.out_proj.bias,
				training=self.training,
				key_padding_mask=key_padding_mask, need_weights=need_weights,
				attn_mask=attn_mask, use_separate_proj_weight=True,
				q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
				v_proj_weight=self.v_proj_weight)
		else:
			if not hasattr(self, '_qkv_same_embed_dim'):
				warnings.warn('A new version of MultiheadAttention module has been implemented. \
					Please re-train your model with the new module',
							  UserWarning)

			return multi_head_attention_forward(
				query, key, value, self.embed_dim, self.num_heads,
				self.in_proj_weight, self.in_proj_bias,
				self.bias_k, self.bias_v, self.add_zero_attn,
				self.dropout, self.out_proj.weight, self.out_proj.bias,
				training=self.training,
				key_padding_mask=key_padding_mask, need_weights=need_weights,
				attn_mask=attn_mask)


def multi_head_attention_forward(query,                           # type: Tensor
								 key,                             # type: Tensor
								 value,                           # type: Tensor
								 embed_dim_to_check,              # type: int
								 num_heads,                       # type: int
								 in_proj_weight,                  # type: Tensor
								 in_proj_bias,                    # type: Tensor
								 bias_k,                          # type: Optional[Tensor]
								 bias_v,                          # type: Optional[Tensor]
								 add_zero_attn,                   # type: bool
								 dropout_p,                       # type: float
								 out_proj_weight,                 # type: Tensor
								 out_proj_bias,                   # type: Tensor
								 training=True,                   # type: bool
								 key_padding_mask=None,           # type: Optional[Tensor]
								 need_weights=True,               # type: bool
								 attn_mask=None,                  # type: Optional[Tensor]
								 use_separate_proj_weight=False,  # type: bool
								 q_proj_weight=None,              # type: Optional[Tensor]
								 k_proj_weight=None,              # type: Optional[Tensor]
								 v_proj_weight=None,              # type: Optional[Tensor]
								 static_k=None,                   # type: Optional[Tensor]
								 static_v=None                    # type: Optional[Tensor]
								 ):
	# type: # (...) -> Tuple[Tensor, Optional[Tensor]]
	r"""
	Args:
		query, key, value: map a query and a set of key-value pairs to an output.
			See "Attention Is All You Need" for more details.
		embed_dim_to_check: total dimension of the model.
		num_heads: parallel attention heads.
		in_proj_weight, in_proj_bias: input projection weight and bias.
		bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
		add_zero_attn: add a new batch of zeros to the key and
					   value sequences at dim=1.
		dropout_p: probability of an element to be zeroed.
		out_proj_weight, out_proj_bias: the output projection weight and bias.
		training: apply dropout if is ``True``.
		key_padding_mask: if provided, specified padding elements in the key will
			be ignored by the attention. This is an binary mask. When the value is True,
			the corresponding value on the attention layer will be filled with -inf.
		need_weights: output attn_output_weights.
		attn_mask: mask that prevents attention to certain positions. This is an additive mask
			(i.e. the values will be added to the attention layer).
		use_separate_proj_weight: the function accept the proj. weights for query, key,
			and value in differnt forms. If false, in_proj_weight will be used, which is
			a combination of q_proj_weight, k_proj_weight, v_proj_weight.
		q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
		static_k, static_v: static key and value used for attention operators.
	Shape:
		Inputs:
		- query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
		  the embedding dimension.
		- key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
		  the embedding dimension.
		- value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
		  the embedding dimension.
		- key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
		- attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
		- static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
		  N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
		- static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
		  N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
		Outputs:
		- attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
		  E is the embedding dimension.
		- attn_output_weights: :math:`(N, L, S)` where N is the batch size,
		  L is the target sequence length, S is the source sequence length.
	"""

	qkv_same = torch.equal(query, key) and torch.equal(key, value)
	kv_same = torch.equal(key, value)

	tgt_len, bsz, embed_dim = query.size()
	assert embed_dim == embed_dim_to_check
	assert list(query.size()) == [tgt_len, bsz, embed_dim]
	assert list(query.size()) == [tgt_len, bsz, embed_dim]
	assert key.size() == value.size()

	head_dim = embed_dim // num_heads
	assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
	scaling = float(head_dim) ** -0.5

	if use_separate_proj_weight is not True:
		if qkv_same:
			# self-attention
			q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

		elif kv_same:
			# encoder-decoder attention
			# This is inline in_proj function with in_proj_weight and in_proj_bias
			_b = in_proj_bias
			_start = 0
			_end = embed_dim
			_w = in_proj_weight[_start:_end, :]
			if _b is not None:
				_b = _b[_start:_end]
			q = linear(query, _w, _b)

			if key is None:
				assert value is None
				k = None
				v = None
			else:

				# This is inline in_proj function with in_proj_weight and in_proj_bias
				_b = in_proj_bias
				_start = embed_dim
				_end = None
				_w = in_proj_weight[_start:, :]
				if _b is not None:
					_b = _b[_start:]
				k, v = linear(key, _w, _b).chunk(2, dim=-1)

		else:
			# This is inline in_proj function with in_proj_weight and in_proj_bias
			_b = in_proj_bias
			_start = 0
			_end = embed_dim
			_w = in_proj_weight[_start:_end, :]
			if _b is not None:
				_b = _b[_start:_end]
			q = linear(query, _w, _b)

			# This is inline in_proj function with in_proj_weight and in_proj_bias
			_b = in_proj_bias
			_start = embed_dim
			_end = embed_dim * 2
			_w = in_proj_weight[_start:_end, :]
			if _b is not None:
				_b = _b[_start:_end]
			k = linear(key, _w, _b)

			# This is inline in_proj function with in_proj_weight and in_proj_bias
			_b = in_proj_bias
			_start = embed_dim * 2
			_end = None
			_w = in_proj_weight[_start:, :]
			if _b is not None:
				_b = _b[_start:]
			v = linear(value, _w, _b)
	else:
		q_proj_weight_non_opt = _unwrap_optional(q_proj_weight)
		len1, len2 = q_proj_weight_non_opt.size()
		assert len1 == embed_dim and len2 == query.size(-1)

		k_proj_weight_non_opt = _unwrap_optional(k_proj_weight)
		len1, len2 = k_proj_weight_non_opt.size()
		assert len1 == embed_dim and len2 == key.size(-1)

		v_proj_weight_non_opt = _unwrap_optional(v_proj_weight)
		len1, len2 = v_proj_weight_non_opt.size()
		assert len1 == embed_dim and len2 == value.size(-1)

		if in_proj_bias is not None:
			q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
			k = linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:(embed_dim * 2)])
			v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
		else:
			q = linear(query, q_proj_weight_non_opt, in_proj_bias)
			k = linear(key, k_proj_weight_non_opt, in_proj_bias)
			v = linear(value, v_proj_weight_non_opt, in_proj_bias)
	q = q * scaling

	if bias_k is not None and bias_v is not None:
		if static_k is None and static_v is None:
			k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
			v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
			if attn_mask is not None:
				attn_mask = torch.cat([attn_mask,
									  torch.zeros((attn_mask.size(0), 1),
												  dtype=attn_mask.dtype,
												  device=attn_mask.device)], dim=1)
			if key_padding_mask is not None:
				key_padding_mask = torch.cat(
					[key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
												   dtype=key_padding_mask.dtype,
												   device=key_padding_mask.device)], dim=1)
		else:
			assert static_k is None, "bias cannot be added to static key."
			assert static_v is None, "bias cannot be added to static value."
	else:
		assert bias_k is None
		assert bias_v is None

	q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
	if k is not None:
		k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
	if v is not None:
		v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

	if static_k is not None:
		assert static_k.size(0) == bsz * num_heads
		assert static_k.size(2) == head_dim
		k = static_k

	if static_v is not None:
		assert static_v.size(0) == bsz * num_heads
		assert static_v.size(2) == head_dim
		v = static_v

	src_len = k.size(1)

	if key_padding_mask is not None:
		assert key_padding_mask.size(0) == bsz
		assert key_padding_mask.size(1) == src_len

	if add_zero_attn:
		src_len += 1
		k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
		v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
		if attn_mask is not None:
			attn_mask = torch.cat([attn_mask, torch.zeros((attn_mask.size(0), 1),
														  dtype=attn_mask.dtype,
														  device=attn_mask.device)], dim=1)
		if key_padding_mask is not None:
			key_padding_mask = torch.cat(
				[key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
											   dtype=key_padding_mask.dtype,
											   device=key_padding_mask.device)], dim=1)

	attn_output_weights = torch.bmm(q, k.transpose(1, 2))
	assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

	if attn_mask is not None:
		attn_mask = attn_mask.unsqueeze(0)
		attn_output_weights += attn_mask

	if key_padding_mask is not None:
		attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
		attn_output_weights = attn_output_weights.masked_fill(
			key_padding_mask.unsqueeze(1).unsqueeze(2),
			float('-inf'),
		)
		attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

	attn_output_weights = softmax(
		attn_output_weights, dim=-1)

	attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)

	attn_output = torch.bmm(attn_output_weights, v)
	assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
	attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
	attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

	if need_weights:
		# average attention weights over heads
		attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
		return attn_output, attn_output_weights.sum(dim=1) / num_heads
	else:
		return attn_output, None

def linear(input, weight, bias=None):
	# type: # (Tensor, Tensor, Optional[Tensor]) -> Tensor
	r"""
	Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.
	Shape:
		- Input: :math:`(N, *, in\_features)` where `*` means any number of
		  additional dimensions
		- Weight: :math:`(out\_features, in\_features)`
		- Bias: :math:`(out\_features)`
		- Output: :math:`(N, *, out\_features)`
	"""
	if input.dim() == 2 and bias is not None:
		# fused op is marginally faster
		ret = torch.addmm(bias, input, weight.t())
	else:
		output = input.matmul(weight.t())
		if bias is not None:
			output += bias
		ret = output
	return ret


def softmax(input, dim=None, _stacklevel=3, dtype=None):
	# type: # (Tensor, Optional[int], int, Optional[int]) -> Tensor
	r"""Applies a softmax function.
	Softmax is defined as:
	:math:`\text{Softmax}(x_{i}) = \frac{exp(x_i)}{\sum_j exp(x_j)}`
	It is applied to all slices along dim, and will re-scale them so that the elements
	lie in the range `[0, 1]` and sum to 1.
	See :class:`~torch.nn.Softmax` for more details.
	Arguments:
		input (Tensor): input
		dim (int): A dimension along which softmax will be computed.
		dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
		  If specified, the input tensor is casted to :attr:`dtype` before the operation
		  is performed. This is useful for preventing data type overflows. Default: None.
	.. note::
		This function doesn't work directly with NLLLoss,
		which expects the Log to be computed between the Softmax and itself.
		Use log_softmax instead (it's faster and has better numerical properties).
	"""
	if dim is None:
		dim = _get_softmax_dim('softmax', input.dim(), _stacklevel)
	if dtype is None:
		ret = input.softmax(dim)
	else:
		ret = input.softmax(dim, dtype=dtype)
	return ret

def _get_softmax_dim(name, ndim, stacklevel):
	# type: # (str, int, int) -> int
	warnings.warn("Implicit dimension choice for {} has been deprecated. "
				  "Change the call to include dim=X as an argument.".format(name), stacklevel=stacklevel)
	if ndim == 0 or ndim == 1 or ndim == 3:
		ret = 0
	else:
		ret = 1
	return ret

# def dropout2(input, p=0.5, training=True, inplace=False):
# 	# type: # (Tensor, float, bool, bool) -> Tensor
# 	r"""
# 	During training, randomly zeroes some of the elements of the input
# 	tensor with probability :attr:`p` using samples from a Bernoulli
# 	distribution.
# 	See :class:`~torch.nn.Dropout` for details.
# 	Args:
# 		p: probability of an element to be zeroed. Default: 0.5
# 		training: apply dropout if is ``True``. Default: ``True``
# 		inplace: If set to ``True``, will do this operation in-place. Default: ``False``
# 	"""
# 	if p < 0. or p > 1.:
# 		raise ValueError("dropout probability has to be between 0 and 1, "
# 						 "but got {}".format(p))
# 	return (_VF.dropout_(input, p, training)
# 			if inplace
# 			else _VF.dropout(input, p, training))

def _unwrap_optional(x):
	assert x is not None, "Unwrapping null optional"
	return x

# def xavier_uniform_(tensor, gain=1.):
#     # type: (Tensor, float) -> Tensor
#     r"""Fills the input `Tensor` with values according to the method
#     described in `Understanding the difficulty of training deep feedforward
#     neural networks` - Glorot, X. & Bengio, Y. (2010), using a uniform
#     distribution. The resulting tensor will have values sampled from
#     :math:`\mathcal{U}(-a, a)` where
#
#     .. math::
#         a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}
#
#     Also known as Glorot initialization.
#
#     Args:
#         tensor: an n-dimensional `torch.Tensor`
#         gain: an optional scaling factor
#
#     Examples:
#         # >>> w = torch.empty(3, 5)
#         # >>> nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
#     """
#     fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
#     std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
#     a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
#
#     return _no_grad_uniform_(tensor, -a, a)