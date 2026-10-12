"""Select independently versioned training and benchmark visual caches."""


def feature_sources(config, *, training=False):
    if training:
        return (config.get('train_feature_root', config.get('feature_root')),
                config.get('train_feature_variant_root', config.get('feature_variant_root')))
    return config.get('feature_root'), config.get('feature_variant_root')


def register_training_caches(model, *specs):
    """Admit validated training caches only for the model's frozen visual weights."""
    if not model.freeze_visual or not model.feature_cache_id:
        raise ValueError('Training caches require verified frozen visual weights')
    if any(spec['vision_sha256'] != model.feature_spec['vision_sha256'] for spec in specs):
        raise ValueError('Training cache visual weights differ from model weights')
    for spec in specs:
        model.feature_cache_ids.update(spec.get('member_cache_ids', [spec['cache_id']]))
