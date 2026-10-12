"""Final attention/residual readout used by the shared decision head."""
import torch
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer


class AttentionOnlyDecoderLayer(Qwen3_5DecoderLayer):
    """The native last attention and residual, without the subsequent MLP branch.

    Existing layer instances are reclassified after loading native weights. This
    preserves native activation checkpointing and the original attention modules.
    The removed MLP and post-attention norm have no registered parameters left.
    """
    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                position_ids=None, past_key_values=None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            position_embeddings=position_embeddings, **kwargs)
        return residual + hidden_states






def adapt_readout(backbone, readout_mode, qk_init='pretrained'):
    """Return optional replacement head and an explicit architecture descriptor."""
    if readout_mode not in ('last_attention',):
        raise ValueError('Unknown nondefault readout mode')
    if qk_init != 'pretrained':
        raise ValueError('Unknown native Q/K initialization')
    text = backbone.language_model
    count = len(text.layers)
    if count < 2 or count != text.config.num_hidden_layers:
        raise ValueError('The decision readout requires a complete native layer stack')
    final = text.layers[-1]
    if final.block_type != 'full_attention':
        raise ValueError('The decision readout requires a final full-attention layer')
    metadata = dict(version='timejev_readout_v1', readout_mode=readout_mode,
                    qk_init=qk_init, original_decoder_layers=count)
    if readout_mode == 'last_attention':
        final.__class__ = AttentionOnlyDecoderLayer
        del final.mlp
        del final.post_attention_layernorm
        metadata.update(executed_decoder_layers=count, final_language_norm=True,
                        final_mlp=False, head='original_pointer')
        return None, metadata
