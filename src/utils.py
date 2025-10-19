import argparse
from omegaconf import OmegaConf

def cli_override(cfg_path: str):
    cfg = OmegaConf.load(cfg_path)
    args = _create_arg_parser()

    # CLI で指定された値だけ上書き
    cli_conf = {k: v for k, v in vars(args).items() if v is not None}

    cfg = OmegaConf.merge(cfg, OmegaConf.create(_dict_from_dotkeys(cli_conf)))
    return cfg

# CLI 引数のパーサーを作成
def _create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data.directory.DATA_DIR", type=str)
    parser.add_argument("--data.data_loader.num_train", type=int)
    parser.add_argument("--data.data_loader.num_val", type=int)
    parser.add_argument("--data.data_loader.num_test", type=int)
    parser.add_argument("--data.data_loader.batch_size", type=int)
    parser.add_argument("--data.data_loader.num_workers", type=int)
    parser.add_argument("--train.learning_rate", type=float)
    parser.add_argument("--train.num_epochs", type=int)
    parser.add_argument("--mlflow.experiment_name", type=str)
    parser.add_argument("--mlflow.run_name", type=str)
    args = parser.parse_args()
    return args

# ドットキーをネスト dict に変換してマージ
def _dict_from_dotkeys(dotdict: dict) -> dict:
    result = {}
    for k, v in dotdict.items():
        keys = k.split(".")
        d = result
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        d[keys[-1]] = v
    return result