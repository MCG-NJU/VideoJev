"""The fixed 35B LLM-LR=2e-6 recipe, extended to VTG and choice questions."""


def validate_recipe(config):
    from .config import normalize_recipe
    config = normalize_recipe(config)
    required = dict(training_backend='qwen36_xtuner_grouped_zero3', freeze_visual=True,
        freeze_visual_connector=True, attention_mode='causal', query_placement='last',
        readout_mode='last_attention', head_dim=512, head_similarity='cosine', head_logit_scale=10.,
        grounding_candidates='timestamp_options', sequence_packing=True,
        global_batch_unit='packed_sequence', global_batch_size=64,
        batch_size=1, epochs=1, seed=20260924, max_sequence_length=147456,
        sample_visual_fraction=.8, router_compute_dtype='float32', moe_router_aux_loss_weight=0,
        mtp_enabled=False, weight_decay=.01, warmup_ratio=.05, grad_clip=1,
        training_tasks=['grounding', 'choice'], input_mode='rgb_dynamic')
    world = config.get('expected_world_size', 0)
    if type(world) is not int or world < 1 or 64 % world or config.get('accumulation') != 64//world:
        raise ValueError('Training world size must divide 64; accumulation must be 64/world_size')
    for key in ('embedding_warmup', 'train_new_token_embeddings', 'feature_root',
                'feature_variant_root', 'new_token_lr'):
        if config.get(key):
            raise ValueError(f'Not part of the requested mixed recipe: {key}')
    if config['training_stage'] == 'head_only':
        required.update(freeze_llm=True, lr=0., head_lr=3e-4, grounding_loss='bce',
            soft_iou_weight=0.)
    elif config['training_stage'] == 'joint':
        required.update(freeze_llm=False, lr=2e-6, head_lr=1e-4, grounding_loss='bce_soft_iou',
            soft_iou_weight=2.)
    else:
        raise ValueError('Unknown training stage')
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f'Mixed 35B recipe requires {key}={expected!r}')


def parameter_groups(model, config):
    groups = []
    for head in (False, True):
        for decay in (True, False):
            parameters = [p for name, p in model.named_parameters() if p.requires_grad
                and name.startswith('head.') == head
                and (len(getattr(p, 'ds_shape', p.shape)) >= 2) == decay]
            if parameters:
                groups.append(dict(params=parameters, lr=config['head_lr'] if head else config['lr'],
                                   weight_decay=config['weight_decay'] if decay else 0.))
    return groups
