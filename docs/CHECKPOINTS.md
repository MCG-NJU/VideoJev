# Checkpoint export

- **4B:** `joint/hf_checkpoint/` is a complete Hugging Face package. Floating-point weights match the base model's dtype (BF16); integer buffers are preserved.
- **35B:** the training export contains task weights. Convert it with the matching base model to include the frozen vision encoder.
- DeepSpeed optimizer/RNG/scheduler recovery stays separate. Head warmup hands off through `head.pt`.

## Convert a task checkpoint

```bash
python scripts/export_hf.py \
  --checkpoint /path/to/final/checkpoint \
  --base /path/to/matching/Qwen-model \
  --out /path/to/new/HF-model \
  --expected-id VERIFIED_FINAL_CHECKPOINT_ID \
  --expected-sha256 VERIFIED_SOURCE_SHA256
```

Both sizes default to grounding threshold `0.25`; use `--decoder-threshold VALUE` to change it.

Exports include safetensors shards/index, configuration, tokenizer, processor and inference code. The exporter checks source identity and tensor round trips, then writes the completed package atomically.

Load with `AutoModel.from_pretrained(..., trust_remote_code=True)`. The 4B training evaluator also accepts older `.pt` checkpoints.

The format follows [XTuner's inference export](https://github.com/InternLM/xtuner/blob/76e705134521eff867b409f3b3451df1c4d8dd36/xtuner/v1/model/base.py#L1529). See [release precision notes](PAPER.md#current-release-defaults).
