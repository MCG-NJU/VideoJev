"""VideoJev configuration with the complete native Qwen backbone configuration."""
from transformers import AutoConfig, PreTrainedConfig


class VideoJevConfig(PreTrainedConfig):
    model_type = 'videojev'
    sub_configs = {'backbone_config': AutoConfig}

    def __init__(self, backbone_config=None, head_dim=512, head_logit_scale=10.,
                 decide_token_id=None, option_end_token_id=None, decoder=None,
                 checkpoint_id=None, **kwargs):
        native = (backbone_config.to_dict() if isinstance(backbone_config, PreTrainedConfig)
                  else dict(backbone_config or {'model_type': 'qwen3_5'}))
        model_type = native.pop('model_type')
        if model_type not in ('qwen3_5', 'qwen3_5_moe'):
            raise ValueError('VideoJev requires a native Qwen3.5 dense or MoE backbone')
        self.backbone_config = AutoConfig.for_model(model_type, **native)
        super().__init__(**kwargs)
        self.head_dim = head_dim
        self.head_logit_scale = head_logit_scale
        self.decide_token_id = decide_token_id
        self.option_end_token_id = option_end_token_id
        self.decoder = dict(threshold=.25, smooth=1, postprocess='round_repair_clamp_v1')
        if decoder is not None:
            self.decoder.update(decoder)
        if not 0 < self.decoder['threshold'] < 1:
            raise ValueError('threshold must be between 0 and 1')
        self.checkpoint_id = checkpoint_id
        self.architectures = ['VideoJevForVideoUnderstanding']
        self.auto_map = {'AutoConfig': 'configuration_videojev.VideoJevConfig',
                         'AutoModel': 'modeling_videojev.VideoJevForVideoUnderstanding'}
        self.tie_word_embeddings = False

    def get_text_config(self, decoder=False, encoder=False):
        return self.backbone_config.get_text_config(decoder=decoder, encoder=encoder)
