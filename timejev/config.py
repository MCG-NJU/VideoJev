"""Resolve portable repository paths without changing input manifests."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolve_paths(value):
    if isinstance(value, dict):
        return {key: resolve_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_paths(item) for item in value]
    if isinstance(value, str) and value.startswith(('data/', 'configs/', 'models/')):
        return str(ROOT / value)
    return value


def normalize_recipe(config):
    config = resolve_paths(config)
    if not isinstance(config, dict):
        return config
    if config.get('base', '').endswith('Qwen3.5-4B'):
        config['base'] = os.environ.get('VIDEOJEV_BASE_4B', config['base'])
    elif config.get('base', '').endswith('Qwen3.6-35B-A3B'):
        config['base'] = os.environ.get('VIDEOJEV_BASE_35B', config['base'])
    return config


def read_config(path):
    return normalize_recipe(json.loads(Path(path).read_text()))
