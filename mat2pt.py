import os
import numpy as np
import h5py
import torch


mat_path = "/mnt/c/Users/onion/Documents/data/m_train.mat" # MATLABファイルのパス
pt_path = "/mnt/c/Users/onion/Documents/processed_data/target/m_train.pt" # 保存先の.ptファイルのパス

# pathのディレクトリ部分が存在しない場合は作成、存在してもエラーにしない
os.makedirs(os.path.dirname(pt_path), exist_ok=True)

# MATLABファイルを読み込み、指定された変数をPyTorchのテンソルに変換して保存
with h5py.File(mat_path, "r") as f:
    keys = list(f.keys())
    data = f[keys]
    tensor = torch.tensor(data)
    torch.save(tensor, pt_path)
