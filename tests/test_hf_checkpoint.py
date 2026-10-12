"""HF export precision, standard loading and original task-input compatibility."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from timejev.hf_checkpoint import (base_weight_dtype, export_tensors,
    load_training_checkpoint, checkpoint_fingerprint)


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32])
def test_export_preserves_input_precision_and_integer_buffers(tmp_path, monkeypatch, dtype):
    from transformers import AutoProcessor
    base = tmp_path/'base'
    base.mkdir()
    save_file({'model.language_model.embed_tokens.weight': torch.ones(3, 2, dtype=dtype)},
              base/'model.safetensors')
    assert base_weight_dtype(base) == dtype
    tokenizer = SimpleNamespace(add_special_tokens=lambda *a, **kw: None,
        convert_tokens_to_ids=lambda token: 1 if token == '<|decide|>' else 2)
    monkeypatch.setattr(AutoProcessor, 'from_pretrained', lambda *a, **kw:
                        SimpleNamespace(tokenizer=tokenizer, save_pretrained=lambda p: None))
    source = {'head.weight': torch.randn(17, 3), 'counter': torch.arange(5)}
    native = dict(model_type='qwen3_5', text_config={}, vision_config={})
    out = tmp_path/'export'
    export_tensors(source.items(), out, base=base, native_config=native,
        checkpoint_id='test', max_shard_bytes=80,
        training_state=dict(config={'base': str(base)}, step=1, checkpoint_id='test'))
    assert json.loads((out/'config.json').read_text())['decoder']['threshold'] == .25
    loaded = load_training_checkpoint(out)
    assert loaded['step'] == 1
    for name, tensor in source.items():
        expected = tensor.to(dtype) if tensor.is_floating_point() else tensor
        assert torch.equal(loaded['model'][name], expected)
        assert loaded['model'][name].dtype == expected.dtype
    assert len(list(out.glob('*.safetensors'))) == 2
    assert not list(out.glob('*.pt'))
    old = checkpoint_fingerprint(out)
    (out/'conversion.json').write_text('{}')
    assert checkpoint_fingerprint(out) != old
    with pytest.raises(FileExistsError):
        export_tensors(source.items(), out, base=base, native_config=native, checkpoint_id='test')


@pytest.fixture(scope='module')
def processor():
    from timejev.decision_inputs import load_processor
    base = Path(os.environ.get('VIDEOJEV_TEST_MODEL', 'models/Qwen3.5-4B'))
    if not base.is_dir():
        pytest.skip('Set VIDEOJEV_TEST_MODEL to a native tokenizer for integration checks')
    return load_processor(str(base))


def test_hf_query_groups_match_training_inputs(processor):
    from test_helpers import record_for
    from timejev.multiturn_inputs import episode_inputs
    from timejev.multiturn_evaluation import query_chunks
    from videojev_hf.processing_videojev import prepare_query_groups
    record = record_for()
    rows = [dict(id=f'q/{i}', task='grounding' if i % 2 == 0 else 'choice',
        query=f'What happens in event {i}?', options=['A. Walking.', 'B. Running.'],
        video_path='/example.mp4', video_key='v', duration=2., spans=[[0., 1.]], answer_index=1)
        for i in range(5)]
    for context in (256, 4096):
        chunks = query_chunks(processor, record, rows, context)
        actual = list(prepare_query_groups(processor, record, rows, context))
        assert len(actual) == len(chunks)
        for (batch, group), expected in zip(actual, chunks):
            assert group == rows[expected['start']:expected['end']]
            original = episode_inputs(processor, record, group)
            for name in ('input_ids', 'mm_token_type_ids', 'video_grid_thw'):
                assert torch.equal(batch[name], original[name])


def test_custom_config_and_automodel_roundtrip(tmp_path):
    import shutil
    from transformers import AutoModel
    from videojev_hf.configuration_videojev import VideoJevConfig
    from videojev_hf.modeling_videojev import VideoJevForVideoUnderstanding
    config = VideoJevConfig(backbone_config=dict(model_type='qwen3_5',
        text_config=dict(vocab_size=32, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            head_dim=16, layer_types=['full_attention']*2,
            rope_parameters=dict(rope_type='default', rope_theta=10000.,
                                 partial_rotary_factor=.5, mrope_section=[1, 1, 2])),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64,
            num_heads=2, out_hidden_size=32, num_position_embeddings=16)),
        head_dim=8)
    config._attn_implementation = 'eager'
    model = VideoJevForVideoUnderstanding(config).eval()
    model.save_pretrained(tmp_path, max_shard_size='1MB')
    for p in (Path(__file__).resolve().parents[1]/'videojev_hf').glob('*.py'):
        if p.name != '__init__.py':
            shutil.copy2(p, tmp_path/p.name)
    loaded, info = AutoModel.from_pretrained(tmp_path, trust_remote_code=True,
        attn_implementation='eager', output_loading_info=True)
    assert not any(info.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'))
    assert set(loaded.state_dict()) == set(model.state_dict())
    assert not hasattr(loaded.backbone.language_model.layers[-1], 'mlp')
    for name, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)


def test_final_export_retry_preserves_previous_attempt(tmp_path):
    from timejev.skip_full_safety import archive_replayed_progress
    completed = tmp_path/'hf_checkpoint'
    completed.mkdir()
    (completed/'model.safetensors').write_bytes(b'previous weights')
    partial = tmp_path/'hf_checkpoint.incomplete'
    partial.mkdir()
    (partial/'partial').write_bytes(b'partial weights')
    (tmp_path/'completion.json').write_text('{"complete": false}')
    archive = archive_replayed_progress(tmp_path, saved_step=0, rank=0, world=1)
    assert (archive/'hf_checkpoint/model.safetensors').read_bytes() == b'previous weights'
    assert (archive/'hf_checkpoint.incomplete/partial').read_bytes() == b'partial weights'
    assert not completed.exists() and not partial.exists()


def test_completed_hf_export_skips_joint_training(tmp_path):
    from scripts.pipeline import action
    joint=tmp_path/'joint'
    (joint/'hf_checkpoint').mkdir(parents=True)
    (joint/'completion.json').write_text('{"complete": true}')
    (joint/'hf_checkpoint/export_complete.json').write_text('{"complete": true}')
    assert action(tmp_path,'joint') == 'skip'
