# Checkpoint export

The default 4B joint-stage artifact is `joint/hf_checkpoint/`. It contains sharded safetensors, a weight index, the complete model configuration, tokenizer/processor, and the VideoJev inference implementation. The language backbone uses native Qwen modules; the final MLP is removed and the trained decision head is included.

The exporter reads the actual embedding-weight dtype from the input model's safetensors. It casts each floating tensor to that dtype, preserves integer buffers, round-trips every written tensor, and records file hashes before atomically publishing the directory. Qwen3.5-4B and Qwen3.6-35B-A3B use BF16. Native nonpersistent RoPE buffers are reconstructed by Transformers and must not be blanket-cast.

DeepSpeed optimizer/master weights, RNG and scheduler state remain in separate recovery checkpoints. The head warmup handoff remains `head.pt`; it is an intermediate training artifact. Resuming an interrupted final export archives the earlier export attempt before rebuilding it from the complete DeepSpeed state.

This follows XTuner's separation between training recovery and inference export, its explicit save dtype, safetensors shards/index, and tokenizer saving. References at the tested revision:

- [Trainer HF checkpoint lifecycle](https://github.com/InternLM/xtuner/blob/76e705134521eff867b409f3b3451df1c4d8dd36/xtuner/v1/train/trainer.py#L1801)
- [Model HF export and explicit casting](https://github.com/InternLM/xtuner/blob/76e705134521eff867b409f3b3451df1c4d8dd36/xtuner/v1/model/base.py#L1529)

To convert an earlier completed task checkpoint:

```bash
python scripts/export_hf.py \
  --checkpoint /path/to/final/checkpoint \
  --base /path/to/matching/Qwen-model \
  --out /path/to/new/HF-model \
  --expected-id VERIFIED_FINAL_CHECKPOINT_ID \
  --expected-sha256 VERIFIED_SOURCE_SHA256 \
  --decoder-threshold 0.15
```

Use threshold 0.35 for the final 35B checkpoint. The 35B converter verifies the frozen-vision base configuration and includes its original visual weights. It preserves task weights and remaps only their module prefix into the standalone HF model. Both exports contain the native Qwen tokenizer and processor plus the two trained decision tokens.

The 4B training evaluator accepts both the new HF training directory and older `.pt` files. A standalone model package converted for distribution uses the model card's `AutoModel` interface. The 35B distributed trainer still keeps its task-only sharded export for ZeRO-3 evaluation; use the conversion command above to create a standalone package with vision.

BF16 serialization rounds FP32 training parameters. Tensor round-trip checks and inference consistency tests compare against the same cast policy; they do not establish unchanged full-benchmark scores relative to FP32 training storage.
