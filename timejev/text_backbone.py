"""Configure activation checkpoint groups for the native causal executor."""
def configure_text_backbone(backbone, attention_mode, segment_size, segment_threshold):
    if attention_mode != 'causal':
        raise ValueError('VideoJev uses native causal attention')
    text = backbone.language_model
    text._timejev_attention_mode = attention_mode
    text._timejev_segment_size = segment_size
    text._timejev_segment_threshold = segment_threshold
