from pathlib import Path

import yaml

_config = None


def get_config():
    global _config
    if _config is None:
        config_path = Path(__file__).parent / "config.yaml"
        with open(config_path, "r") as f:
            _config = yaml.safe_load(f)
    return _config
