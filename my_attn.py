from diffusers.models.attention_processor import Attention
import torch
from typing import Optional, Callable
import torch.nn.functional as F
import xformers


class MyAttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        residual = hidden_states

        args = (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)
        self.guidance = key.shape[0] > 1
        
        query = attn.head_to_batch_dim(query)
        if hasattr(self, 'my_query'):
            if self.guidance:
                uncond, cond = torch.chunk(query, 2, dim=0)
                query = torch.cat([uncond, self.my_query], dim=0)
            else:
                query = self.my_query
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        if hasattr(self, 'attn_map'):
            if self.guidance:
                uncond, cond = torch.chunk(attention_probs, 2, dim=0)
                attention_probs = torch.cat([uncond, self.attn_map], dim=0)
            else:
                attention_probs = self.attn_map
        
        self.key = key
        self.query = query
        self.value = value
        self.attention_probs = attention_probs
        
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
    
    
    def record_QKV(self):
        return self.query, self.key, self.value, self.attention_probs


class MyAttnProcessor2:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = (scale,)
        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        # Record the Q,K,V for PCA guidance
        self.key = key
        self.query = query
        self.value = value
        # self.hidden_state = hidden_states.detach()

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        # hidden_states = F.scaled_dot_product_attention(
        #     query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        # )
        
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        self.attention_probs = attention_probs
        hidden_states = torch.bmm(attention_probs, value)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

def prep_unet_attention(unet):
    # replace the fwd function
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "Attention":
            module.set_processor(MyAttnProcessor())


# class MySelfAttnProcessor:
#     def __init__(self, attention_op: Optional[Callable] = None):
#         self.attention_op = attention_op

#     def __call__(
#         self,
#         attn: Attention,
#         hidden_states,
#         encoder_hidden_states=None,
#         attention_mask=None,
#         temb=None,
#         scale: float = 1.0
#     ):

#         residual = hidden_states

#         args = (scale,)

#         if attn.spatial_norm is not None:
#             hidden_states = attn.spatial_norm(hidden_states, temb)

#         input_ndim = hidden_states.ndim

#         if input_ndim == 4:
#             batch_size, channel, height, width = hidden_states.shape
#             hidden_states = hidden_states.view(
#                 batch_size, channel, height * width).transpose(1, 2)

#         batch_size, key_tokens, _ = (
#             hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
#         )

#         attention_mask = attn.prepare_attention_mask(
#             attention_mask, key_tokens, batch_size)
#         self.attention_mask = attention_mask
#         self.attn = attn
#         if attention_mask is not None:
#             # expand our mask's singleton query_tokens dimension:
#             #   [batch*heads,            1, key_tokens] ->
#             #   [batch*heads, query_tokens, key_tokens]
#             # so that it can be added as a bias onto the attention scores that xformers computes:
#             #   [batch*heads, query_tokens, key_tokens]
#             # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
#             _, query_tokens, _ = hidden_states.shape
#             attention_mask = attention_mask.expand(-1, query_tokens, -1)

#         if attn.group_norm is not None:
#             hidden_states = attn.group_norm(
#                 hidden_states.transpose(1, 2)).transpose(1, 2)

#         query = attn.to_q(hidden_states, scale=scale)

#         if encoder_hidden_states is None:
#             encoder_hidden_states = hidden_states
#         elif attn.norm_cross:
#             encoder_hidden_states = attn.norm_encoder_hidden_states(
#                 encoder_hidden_states)

#         key = attn.to_k(encoder_hidden_states, scale=scale)
#         value = attn.to_v(encoder_hidden_states, scale=scale)

#         # Record the Q,K,V for PCA guidance
#         self.key = key
#         self.query = query
#         self.value = value
#         self.hidden_state = hidden_states.detach()

#         query = attn.head_to_batch_dim(query).contiguous()
#         key = attn.head_to_batch_dim(key).contiguous()
#         value = attn.head_to_batch_dim(value).contiguous()

#         hidden_states = xformers.ops.memory_efficient_attention(
#             query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
#         )
#         hidden_states = hidden_states.to(query.dtype)
#         hidden_states = attn.batch_to_head_dim(hidden_states)

#         # linear proj
#         hidden_states = attn.to_out[0](hidden_states)
#         # dropout
#         hidden_states = attn.to_out[1](hidden_states)

#         if input_ndim == 4:
#             hidden_states = hidden_states.transpose(
#                 -1, -2).reshape(batch_size, channel, height, width)

#         if attn.residual_connection:
#             hidden_states = hidden_states + residual

#         hidden_states = hidden_states / attn.rescale_output_factor

#         return hidden_states
