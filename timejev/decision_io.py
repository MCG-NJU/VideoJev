"""The final timestamp-option protocol serves both VTG and MCQ."""
def data_module(config):
    if config.get('grounding_candidates') != 'timestamp_options':
        raise ValueError('Expected timestamp-option protocol')
    from . import timestamp_option_data
    return timestamp_option_data
