"""Load the released ModelScope checkpoint and run direct video decisions."""

import os
from functools import lru_cache
from pathlib import Path

MODEL_ID = 'MCG-NJU/VideoJev-4B'
MODEL_REVISION = '5bff12c30ef31175cf6e2c07dd01ecb71b28d36a'
FPS = 2.0
MAX_FRAMES = 2048


@lru_cache(maxsize=1)
def load_model():
    import torch
    from transformers import AutoModel, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError('This demo needs a CUDA GPU. Switch the Studio to GPU hardware.')

    local = os.environ.get('VIDEOJEV_CHECKPOINT')
    if local:
        checkpoint = str(Path(local).expanduser().resolve())
        if not Path(checkpoint, 'config.json').is_file():
            raise FileNotFoundError(f'No VideoJev checkpoint found at {checkpoint}')
    else:
        from modelscope import snapshot_download

        checkpoint = snapshot_download(MODEL_ID, revision=MODEL_REVISION)

    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={'': 'cuda:0'},
        attn_implementation='flash_attention_2',
    )
    model.prepare_for_inference()
    return model, processor


def predict(video_path, questions, threshold, context):
    model, processor = load_model()
    return model.predict_video(
        processor,
        video_path,
        questions,
        fps=FPS,
        max_frames=MAX_FRAMES,
        context=context,
        threshold=threshold,
    )
