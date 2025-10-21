import mlflow
from omegaconf import OmegaConf


def log_config_to_mlflow(cfg, parent_key=""):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)  # Dict型に変換（resolve=Trueで値を展開）

    def _recursive_log(d, prefix=""):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                _recursive_log(v, prefix=key)
            else:
                mlflow.log_param(key, v)

    _recursive_log(cfg_dict, parent_key)